import cv2
from ultralytics import YOLO
from collections import Counter, deque
from classifier import classify_passenger

# =========================
# LOAD MODELS
# =========================

rickshaw_model = YOLO("../models/rickshaw_detector.pt")
person_model = YOLO("yolov8m.pt")

# =========================
# VIDEO SETUP
# =========================

video_path = "../videos/test_videos3.mp4"
output_path = "../outputs/main_pipeline_phase3_final.mp4"

cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# =========================
# MEMORY
# =========================

status_memory = {}
count_memory = {}
label_memory = {}

# =========================
# PHASE 2 MEMORY
# =========================

rickshaw_memory = {}

MEMORY_SIZE = 5

# =========================
# HELPER
# =========================

def is_duplicate(box1, box2):
    x1, y1, x2, y2 = box1
    x1f, y1f, x2f, y2f = box2

    inter_w = max(0, min(x2, x2f) - max(x1, x1f))
    inter_h = max(0, min(y2, y2f) - max(y1, y1f))
    inter_area = inter_w * inter_h

    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2f - x1f) * (y2f - y1f)

    iou = inter_area / (area1 + area2 - inter_area + 1e-6)

    return iou > 0.6


# =========================
# MAIN LOOP
# =========================

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    results = rickshaw_model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml"
    )

    if results[0].boxes.id is None:
        out.write(frame)
        continue

    boxes = results[0].boxes.xyxy
    ids = results[0].boxes.id

    for i, box in enumerate(boxes):

        track_id = int(ids[i])
        x1, y1, x2, y2 = map(int, box)

        w = x2 - x1
        h = y2 - y1

        # =========================
        # CROP
        # =========================

        x1s = int(x1 + w * 0.10)
        x2s = int(x2 - w * 0.10)

        y1s = int(y1 + h * 0.05)
        y2s = int(y2 - h * 0.05)

        x1s = max(0, x1s)
        y1s = max(0, y1s)
        x2s = min(width, x2s)
        y2s = min(height, y2s)

        crop = frame[y1s:y2s, x1s:x2s]

        if crop.size == 0:
            continue

        # =========================
        # PERSON DETECTION
        # =========================

        person_results = person_model(crop)

        person_centers = []
        person_boxes = []

        for j, cls in enumerate(person_results[0].boxes.cls):

            if int(cls) != 0:
                continue

            conf = float(person_results[0].boxes.conf[j])
            if conf < 0.5:
                continue

            px1, py1, px2, py2 = map(int, person_results[0].boxes.xyxy[j])

            if (py2 - py1) < 18:
                continue

            cx = int((px1 + px2) / 2)
            cy = int((py1 + py2) / 2)

            if cx < crop.shape[1] * 0.20 or cx > crop.shape[1] * 0.80:
                continue

            if cy > crop.shape[0] * 0.8:
                continue

            person_centers.append((cx, cy))
            person_boxes.append((px1, py1, px2, py2))

        # =========================
        # DUPLICATE REMOVAL
        # =========================

        filtered_boxes = []
        filtered_centers = []

        for idx, (px1, py1, px2, py2) in enumerate(person_boxes):

            keep = True

            for fbox in filtered_boxes:
                if is_duplicate((px1, py1, px2, py2), fbox):
                    keep = False
                    break

            if keep:
                filtered_boxes.append((px1, py1, px2, py2))
                filtered_centers.append(person_centers[idx])

        person_boxes = filtered_boxes
        person_centers = filtered_centers

        # =========================
        # PASSENGER REGION FILTER
        # =========================

        valid_centers = []
        valid_boxes = []

        for idx, (cx, cy) in enumerate(person_centers):

            if cy < crop.shape[0] * 0.65:
                valid_centers.append((cx, cy))
                valid_boxes.append(person_boxes[idx])

        person_centers = valid_centers
        person_boxes = valid_boxes

        # =========================
        # PASSENGER COUNT
        # =========================

        passenger_count = 0
        passenger_boxes = []

        if len(person_centers) > 0:

            driver = max(person_centers, key=lambda p: p[1])

            for idx, p in enumerate(person_centers):
                if p != driver:
                    passenger_count += 1
                    passenger_boxes.append(person_boxes[idx])

        # =========================
        # PHASE 2 - TEMPORAL MEMORY
        # =========================

        if track_id not in rickshaw_memory:
            rickshaw_memory[track_id] = deque(maxlen=MEMORY_SIZE)

        rickshaw_memory[track_id].append(passenger_count)

        history = rickshaw_memory[track_id]
        temporal_count = round(sum(history) / len(history))

        # Combine both
        if track_id not in count_memory:
            count_memory[track_id] = []

        count_memory[track_id].append(passenger_count)
        count_memory[track_id] = count_memory[track_id][-10:]

        smooth_count = Counter(count_memory[track_id]).most_common(1)[0][0]

        final_count = max(smooth_count, temporal_count)

        # =========================
        # PHASE 3 - GENDER STABILITY
        # =========================

        passenger_labels = []

        for (px1, py1, px2, py2) in passenger_boxes:

            passenger_crop = crop[py1:py2, px1:px2]

            if passenger_crop.size == 0:
                continue

            # label, conf = classify_passenger(passenger_crop)

            result = classify_passenger(passenger_crop)

            # handle both cases safely
            if isinstance(result, tuple):
                label, conf = result
            else:
                label = result
                conf = 1.0  # assume high confidence

            if conf < 0.6:
                continue

            cx = int((px1 + px2) / 2)
            cy = int((py1 + py2) / 2)

            key = (track_id, int(cx / 20), int(cy / 20))

            if key not in label_memory:
                label_memory[key] = []

            label_memory[key].append(label)
            label_memory[key] = label_memory[key][-10:]

            counts = Counter(label_memory[key])
            total = sum(counts.values())

            final_label, freq = counts.most_common(1)[0]

            if freq / total >= 0.6:
                passenger_labels.append(final_label)

        passenger_labels = passenger_labels[:final_count]

        # =========================
        # VACANCY LOGIC (UNCHANGED ✅)
        # =========================

        if final_count >= 1:
            status = "Occupied"
        else:
            status = "Vacant"

        if track_id not in status_memory:
            status_memory[track_id] = []

        status_memory[track_id].append(status)
        status_memory[track_id] = status_memory[track_id][-10:]

        if status_memory[track_id].count("Occupied") > 5:
            final_status = "Occupied"
            color = (0, 255, 0)
        else:
            final_status = "Vacant"
            color = (0, 0, 255)

        # =========================
        # DISPLAY
        # =========================

        label_text = f"{final_status} | {final_count}"

        if passenger_labels:
            label_text += f" ({', '.join(passenger_labels)})"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            frame,
            label_text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    out.write(frame)

    cv2.imshow("FINAL PIPELINE", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()