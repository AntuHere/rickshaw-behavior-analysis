import cv2
from ultralytics import YOLO
from collections import defaultdict, deque
import math
import os

# =========================
# OUTPUT
# =========================
SCRIPT_NAME = "negotiation_2"

video_path = "../../videos/test_videos3.mp4"
video_name = os.path.basename(video_path).split(".")[0]

base_output_dir = "../../outputs"
script_dir = os.path.join(base_output_dir, SCRIPT_NAME)
os.makedirs(script_dir, exist_ok=True)

version = 1
while True:
    output_path = os.path.join(
        script_dir,
        f"{video_name}_Nego_{version}.mp4"
    )

    if not os.path.exists(output_path):
        break
    version += 1

print(f"[INFO] Saving to: {output_path}")

# =========================
# MODELS
# =========================
rickshaw_model = YOLO("../../models/rickshaw_detector.pt")
person_model = YOLO("yolov8m.pt")

# =========================
# VIDEO
# =========================
cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

# =========================
# PARAMETERS
# =========================
OUTER_SCALE = 1.35

STOP_THRESHOLD = 1.5
VAR_THRESHOLD = 14
DISP_THRESHOLD = 10

STOP_ENTER = 8
STOP_EXIT = 3

VISIBLE_MIN_FRAMES = 2
DISAPPEAR_FRAMES = 2
CLEANUP_FRAMES = 8

EVENT_COOLDOWN = 20
STATE_HOLD = 20

NEGOTIATION_FRAMES = 20
FAILED_NEGOTIATION_FRAMES = int(fps * 1.5)

# =========================
# MEMORY
# =========================
motion_mem = defaultdict(lambda: deque(maxlen=12))
stop_counter = defaultdict(int)

hist_pos = defaultdict(lambda: deque(maxlen=12))
visible_count = defaultdict(int)
missing_count = defaultdict(int)
inner_hits = defaultdict(int)

event_cooldown = defaultdict(int)
rickshaw_state = defaultdict(lambda: {"state": "", "timer": 0})
rickshaw_occupied = defaultdict(bool)

# NEW
negotiation_active = defaultdict(bool)

frame_id = 0

# =========================
# HELPERS
# =========================
def expand_box(x1, y1, x2, y2, scale):
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2)//2, (y1 + y2)//2

    return (
        int(cx - w*scale/2),
        int(cy - h*scale/2),
        int(cx + w*scale/2),
        int(cy + h*scale/2)
    )


def inside(box, pt):
    x1, y1, x2, y2 = box
    return x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2


def center(box):
    x1, y1, x2, y2 = box
    return ((x1+x2)//2, (y1+y2)//2)


def dist(p, q):
    return math.hypot(p[0]-q[0], p[1]-q[1])


def avg_speed(mem):
    if len(mem) < 2:
        return 0

    pts = list(mem)

    return sum(
        dist(pts[i], pts[i-1])
        for i in range(1, len(pts))
    ) / len(pts)


def position_variation(mem):
    if len(mem) < 5:
        return 999

    pts = list(mem)

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    return (max(xs)-min(xs)) + (max(ys)-min(ys))


def displacement(mem):
    if len(mem) < 6:
        return 999

    pts = list(mem)
    return dist(pts[0], pts[-1])


def draw_box(frame, box, label, color):
    x1, y1, x2, y2 = box

    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

    (w,h), _ = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        2
    )

    cv2.rectangle(
        frame,
        (x1, y1-25),
        (x1+w, y1),
        color,
        -1
    )

    cv2.putText(
        frame,
        label,
        (x1, y1-5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255,255,255),
        2
    )


# =========================
# MAIN LOOP
# =========================
while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    r = rickshaw_model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml"
    )

    p = person_model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml"
    )

    if r[0].boxes.id is None:
        out.write(frame)
        continue

    r_boxes = r[0].boxes.xyxy
    r_ids = r[0].boxes.id

    persons = []

    if p[0].boxes.id is not None:
        for j, pb in enumerate(p[0].boxes.xyxy):
            px1, py1, px2, py2 = map(int, pb)

            persons.append((
                int(p[0].boxes.id[j]),
                ((px1+px2)//2, (py1+py2)//2)
            ))

    assignments = defaultdict(list)

    r_centers = {
        int(r_ids[i]): center(tuple(map(int, r_boxes[i])))
        for i in range(len(r_boxes))
    }

    for pid, pc in persons:

        candidates = []

        for i, rb in enumerate(r_boxes):
            r_id = int(r_ids[i])

            outer = expand_box(
                *map(int, rb),
                OUTER_SCALE
            )

            if inside(outer, pc):
                candidates.append(
                    (r_id, dist(pc, r_centers[r_id]))
                )

        if candidates:
            r_id = min(candidates, key=lambda x:x[1])[0]
            assignments[r_id].append((pid, pc))

    # =========================
    # PROCESS EACH RICKSHAW
    # =========================
    for i, rb in enumerate(r_boxes):

        r_id = int(r_ids[i])

        x1, y1, x2, y2 = map(int, rb)

        # actual passenger seat zone only
        seat_x1 = x1 + int((x2 - x1) * 0.22)
        seat_x2 = x2 - int((x2 - x1) * 0.22)

        seat_y1 = y1 + int((y2 - y1) * 0.20)
        seat_y2 = y2 - int((y2 - y1) * 0.28)

        inner = (
            seat_x1,
            seat_y1,
            seat_x2,
            seat_y2
        )
        # =========================
        # DEBUG BOARDING ZONE
        # Remove later after testing
        # =========================
        cv2.rectangle(
            frame,
            (seat_x1, seat_y1),
            (seat_x2, seat_y2),
            (255, 0, 0),
            2
        )
        rc = center((x1, y1, x2, y2))


        # -------------------------
        # STOP DETECTION
        # -------------------------
        prev_stop = stop_counter[r_id] >= STOP_ENTER

        motion_mem[r_id].append(rc)

        sp = avg_speed(motion_mem[r_id])
        var = position_variation(motion_mem[r_id])
        disp = displacement(motion_mem[r_id])

        if (
            sp < STOP_THRESHOLD
            and var < VAR_THRESHOLD
            and disp < DISP_THRESHOLD
        ):
            stop_counter[r_id] += 1
        else:
            stop_counter[r_id] = max(0, stop_counter[r_id]-1)

        if stop_counter[r_id] >= STOP_ENTER:
            is_stopped = True
        elif stop_counter[r_id] <= STOP_EXIT:
            is_stopped = False
        else:
            is_stopped = prev_stop

        # Rickshaw moved away → stop negotiation
        if not is_stopped:
            negotiation_active[r_id] = False

        if event_cooldown[r_id] > 0:
            event_cooldown[r_id] -= 1

        current_keys = set()

        # =========================
        # PERSON TRACKING
        # =========================
        for pid, pc in assignments[r_id]:

            key = (pid, r_id)
            current_keys.add(key)

            hist_pos[key].append(pc)
            visible_count[key] += 1
            missing_count[key] = 0

            if inside(inner, pc):
                inner_hits[key] += 1

            # -------------------------
            # NEGOTIATION START
            # -------------------------
            if (
                visible_count[key] >= NEGOTIATION_FRAMES
                and inner_hits[key] == 0
                and is_stopped
                and not negotiation_active[r_id]
            ):
                negotiation_active[r_id] = True

        # =========================
        # BOARDING / FAILED / UNBOARDING
        # =========================
        for key in list(hist_pos.keys()):

            pid_k, rid_k = key

            if rid_k != r_id:
                continue

            if key not in current_keys:

                missing_count[key] += 1

                # FAILED NEGOTIATION
                if (
                    negotiation_active[r_id]
                    and inner_hits[key] == 0
                    and missing_count[key] >= FAILED_NEGOTIATION_FRAMES
                ):
                    rickshaw_state[r_id] = {
                        "state": "Failed Negotiation",
                        "timer": STATE_HOLD
                    }

                    negotiation_active[r_id] = False

                # BOARDING
                if (
                    visible_count[key] >= VISIBLE_MIN_FRAMES
                    and inner_hits[key] >= 1
                    and missing_count[key] >= DISAPPEAR_FRAMES
                    and is_stopped
                    and event_cooldown[r_id] == 0
                ):
                    rickshaw_state[r_id] = {
                        "state": "Boarding",
                        "timer": STATE_HOLD
                    }

                    rickshaw_occupied[r_id] = True
                    negotiation_active[r_id] = False
                    event_cooldown[r_id] = EVENT_COOLDOWN

                # UNBOARDING
                if (
                    rickshaw_occupied[r_id]
                    and inner_hits[key] >= 1
                    and missing_count[key] >= DISAPPEAR_FRAMES
                    and is_stopped
                    and event_cooldown[r_id] == 0
                ):
                    rickshaw_state[r_id] = {
                        "state": "Unboarding",
                        "timer": STATE_HOLD
                    }

                    rickshaw_occupied[r_id] = False
                    event_cooldown[r_id] = EVENT_COOLDOWN

                # CLEANUP
                if missing_count[key] > CLEANUP_FRAMES:
                    hist_pos.pop(key, None)
                    visible_count.pop(key, None)
                    missing_count.pop(key, None)
                    inner_hits.pop(key, None)

        # =========================
        # DISPLAY LOGIC
        # =========================
        display = ""

        if negotiation_active[r_id]:
            display = "Negotiating"

        elif rickshaw_state[r_id]["timer"] > 0:
            display = rickshaw_state[r_id]["state"]
            rickshaw_state[r_id]["timer"] -= 1

        color = (0,0,255) if is_stopped else (0,255,0)

        label = "Stopped" if is_stopped else "Moving"

        if display:
            label += f" | {display}"

        draw_box(
            frame,
            (x1,y1,x2,y2),
            label,
            color
        )

    out.write(frame)
    cv2.imshow("NEGOTIATION SYSTEM", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()