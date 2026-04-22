import cv2
from ultralytics import YOLO
from collections import defaultdict, deque
import math
import os
from datetime import datetime

# =========================
# OUTPUT
# =========================
SCRIPT_NAME = "phase_4_FinalBoardingVersion"

base_output_dir = "../outputs"
script_dir = os.path.join(base_output_dir, SCRIPT_NAME)

run_name = datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")
run_dir = os.path.join(script_dir, run_name)
os.makedirs(run_dir, exist_ok=True)

video_path = "../videos/test_videos3.mp4"
video_name = os.path.basename(video_path).split(".")[0]
output_path = os.path.join(run_dir, f"{video_name}_output.mp4")

print(f"[INFO] Saving to: {output_path}")

# =========================
# MODELS
# =========================
rickshaw_model = YOLO("../models/rickshaw_detector.pt")
person_model   = YOLO("yolov8m.pt")

# =========================
# VIDEO
# =========================
cap = cv2.VideoCapture(video_path)

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(output_path,
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps,
                      (width, height))

# =========================
# PARAMETERS
# =========================
OUTER_SCALE = 1.35

STOP_THRESHOLD = 1.5
VAR_THRESHOLD  = 14
DISP_THRESHOLD = 10

STOP_ENTER = 8
STOP_EXIT  = 3

VISIBLE_MIN_FRAMES = 2
DISAPPEAR_FRAMES   = 2
CLEANUP_FRAMES     = 8

APPROACH_FRAMES = 4
MOVE_THRESHOLD  = 10

EVENT_COOLDOWN = 20
STATE_HOLD     = 20

# =========================
# MEMORY
# =========================
motion_mem   = defaultdict(lambda: deque(maxlen=12))
stop_counter = defaultdict(int)

hist_pos      = defaultdict(lambda: deque(maxlen=12))
visible_count = defaultdict(int)
missing_count = defaultdict(int)
inner_hits    = defaultdict(int)

event_cooldown = defaultdict(int)
rickshaw_state = defaultdict(lambda: {"state": "", "timer": 0})
rickshaw_occupied = defaultdict(lambda: False)

stop_start_frame = defaultdict(int)
frame_id = 0

# =========================
# HELPERS
# =========================
def expand_box(x1,y1,x2,y2,scale):
    w,h=x2-x1,y2-y1
    cx,cy=(x1+x2)//2,(y1+y2)//2
    return int(cx-w*scale/2),int(cy-h*scale/2),int(cx+w*scale/2),int(cy+h*scale/2)

def inside(box,pt):
    x1,y1,x2,y2=box
    return x1<=pt[0]<=x2 and y1<=pt[1]<=y2

def center(box):
    x1,y1,x2,y2=box
    return ((x1+x2)//2,(y1+y2)//2)

def dist(p,q):
    return math.hypot(p[0]-q[0],p[1]-q[1])

def avg_speed(mem):
    if len(mem)<2: return 0
    pts=list(mem)
    return sum(dist(pts[i],pts[i-1]) for i in range(1,len(pts)))/len(pts)

def position_variation(mem):
    if len(mem)<5: return 999
    pts=list(mem)
    xs=[p[0] for p in pts]
    ys=[p[1] for p in pts]
    return (max(xs)-min(xs))+(max(ys)-min(ys))

def displacement(mem):
    if len(mem)<6: return 999
    pts=list(mem)
    return dist(pts[0], pts[-1])

def draw_box(frame, box, label, color):
    x1,y1,x2,y2 = box
    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
    (w,h),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.6,2)
    cv2.rectangle(frame,(x1,y1-25),(x1+w,y1),color,-1)
    cv2.putText(frame,label,(x1,y1-5),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)



# =========================
# MAIN LOOP
# =========================
while cap.isOpened():

    ret,frame=cap.read()
    if not ret: break
    frame_id+=1

    r=rickshaw_model.track(frame,persist=True,tracker="bytetrack.yaml")
    p=person_model.track(frame,persist=True,tracker="bytetrack.yaml")

    if r[0].boxes.id is None:
        out.write(frame)
        continue

    r_boxes=r[0].boxes.xyxy
    r_ids=r[0].boxes.id

    persons=[]
    if p[0].boxes.id is not None:
        for j,pb in enumerate(p[0].boxes.xyxy):
            px1,py1,px2,py2=map(int,pb)
            persons.append((int(p[0].boxes.id[j]),
                            ((px1+px2)//2,(py1+py2)//2)))

    assignments=defaultdict(list)
    r_centers={int(r_ids[i]):center(tuple(map(int,r_boxes[i]))) for i in range(len(r_boxes))}

    for pid,pc in persons:
        candidates=[]
        for i,rb in enumerate(r_boxes):
            r_id=int(r_ids[i])
            outer=expand_box(*map(int,rb),OUTER_SCALE)
            if inside(outer,pc):
                candidates.append((r_id,dist(pc,r_centers[r_id])))
        if candidates:
            r_id=min(candidates,key=lambda x:x[1])[0]
            assignments[r_id].append((pid,pc))

    for i,rb in enumerate(r_boxes):

        r_id=int(r_ids[i])
        x1,y1,x2,y2=map(int,rb)
        inner=(x1,y1,x2,y2)
        rc=center(inner)

        prev_stop=stop_counter[r_id]>=STOP_ENTER

        motion_mem[r_id].append(rc)
        sp=avg_speed(motion_mem[r_id])
        var=position_variation(motion_mem[r_id])
        disp=displacement(motion_mem[r_id])

        if sp < STOP_THRESHOLD and var < VAR_THRESHOLD and disp < DISP_THRESHOLD:
            stop_counter[r_id]+=1
        else:
            stop_counter[r_id]=max(0,stop_counter[r_id]-1)

        if stop_counter[r_id]>=STOP_ENTER: is_stopped=True
        elif stop_counter[r_id]<=STOP_EXIT: is_stopped=False
        else: is_stopped=prev_stop

        if is_stopped and not prev_stop:
            stop_start_frame[r_id]=frame_id

        if event_cooldown[r_id]>0:
            event_cooldown[r_id]-=1

        current_keys=set()

        for pid,pc in assignments[r_id]:
            key=(pid,r_id)
            current_keys.add(key)

            hist_pos[key].append(pc)
            visible_count[key]+=1
            missing_count[key]=0

            if inside(inner,pc):
                inner_hits[key]+=1

        for key in list(hist_pos.keys()):

            pid_k,rid_k=key
            if rid_k!=r_id: continue

            if key not in current_keys:
                missing_count[key]+=1

                # ===== BOARDING ONLY =====
                if (
                    visible_count[key] >= VISIBLE_MIN_FRAMES and
                    missing_count[key] >= DISAPPEAR_FRAMES and
                    is_stopped and
                    event_cooldown[r_id] == 0
                ):
                    rickshaw_state[r_id]={"state":"Boarding","timer":STATE_HOLD}
                    rickshaw_occupied[r_id]=True
                    event_cooldown[r_id]=EVENT_COOLDOWN

                # ===== UNBOARDING ONLY =====
                if (
                    rickshaw_occupied[r_id] and
                    inner_hits[key]>=1 and
                    missing_count[key]>=DISAPPEAR_FRAMES and
                    is_stopped and
                    event_cooldown[r_id]==0
                ):
                    rickshaw_state[r_id]={"state":"Unboarding","timer":STATE_HOLD}
                    rickshaw_occupied[r_id]=False
                    event_cooldown[r_id]=EVENT_COOLDOWN

                if missing_count[key]>CLEANUP_FRAMES:
                    hist_pos.pop(key,None)
                    visible_count.pop(key,None)
                    missing_count.pop(key,None)
                    inner_hits.pop(key,None)

        # =========================
        # UI (FIXED)
        # =========================

        if rickshaw_state[r_id]["timer"] > 0:
            display = rickshaw_state[r_id]["state"]
            rickshaw_state[r_id]["timer"] -= 1
        else:
            display = ""

        # 🔥 COLOR FIX
        # stopped = RED, moving = GREEN
        color = (0, 0, 255) if is_stopped else (0, 255, 0)

        # 🔥 LABEL FIX
        if is_stopped:
            label = "Stopped"
        else:
            label = "Moving"

        if display:
            label += f" | {display}"

        draw_box(frame, (x1, y1, x2, y2), label, color)

    out.write(frame)
    cv2.imshow("CLEAN SYSTEM",frame)

    if cv2.waitKey(1)==27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()