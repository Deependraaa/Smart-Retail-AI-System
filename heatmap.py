from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np

# -----------------------------
# LOAD YOLO MODEL
# -----------------------------
model = YOLO("yolov8n.pt")

# -----------------------------
# INITIALIZE TRACKER
# -----------------------------
tracker = DeepSort(max_age=30)

# -----------------------------
# LOAD VIDEO
# -----------------------------
cap = cv2.VideoCapture("mall.mp4")

# Read first frame
ret, frame = cap.read()

if not ret:
    print("Error reading video")
    exit()

# Get frame dimensions
height, width, _ = frame.shape

# Create empty heatmap
heatmap = np.zeros((height, width), dtype=np.float32)

while True:

    # -----------------------------
    # HEAT DECAY (IMPORTANT)
    # Slowly fade old heat
    # -----------------------------
    heatmap *= 0.95

    # Read video frame
    ret, frame = cap.read()

    if not ret:
        break

    # -----------------------------
    # YOLO DETECTION
    # -----------------------------
    results = model(frame)

    detections = []

    for result in results:

        boxes = result.boxes

        for box in boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # Detect only persons
            if cls == 0 and conf > 0.5:

                x1, y1, x2, y2 = box.xyxy[0]

                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                w = x2 - x1
                h = y2 - y1

                detections.append(([x1, y1, w, h], conf, 'person'))

    # -----------------------------
    # UPDATE TRACKER
    # -----------------------------
    tracks = tracker.update_tracks(detections, frame=frame)

    people_count = 0

    for track in tracks:

        if not track.is_confirmed():
            continue

        people_count += 1

        # Tracking ID
        track_id = track.track_id

        # Get bounding box
        ltrb = track.to_ltrb()

        x1, y1, x2, y2 = map(int, ltrb)

        # -----------------------------
        # DRAW TRACKING BOX
        # -----------------------------
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0,255,0),
            2
        )

        # -----------------------------
        # SHOW TRACKING ID
        # -----------------------------
        cv2.putText(
            frame,
            f"ID: {track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )

        # -----------------------------
        # HEATMAP LOGIC
        # -----------------------------

        # Get center point
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        # Add heat at movement position
        cv2.circle(
            heatmap,
            (center_x, center_y),
            30,
            1,
            -1
        )

    # -----------------------------
    # SMOOTH HEATMAP
    # -----------------------------
    heatmap_blur = cv2.GaussianBlur(
        heatmap,
        (51,51),
        0
    )

    # -----------------------------
    # NORMALIZE HEATMAP
    # -----------------------------
    heatmap_normalized = cv2.normalize(
        heatmap_blur,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    heatmap_uint8 = np.uint8(heatmap_normalized)

    # -----------------------------
    # APPLY HEAT COLORS
    # -----------------------------
    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    # -----------------------------
    # OVERLAY HEATMAP
    # -----------------------------
    overlay = cv2.addWeighted(
        frame,
        0.7,
        heatmap_color,
        0.3,
        0
    )

    # -----------------------------
    # SHOW PEOPLE COUNT
    # -----------------------------
    cv2.putText(
        overlay,
        f"People Count: {people_count}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,255),
        2
    )

    # -----------------------------
    # SAVE HEATMAP IMAGE
    # For Flask Dashboard
    # -----------------------------
    cv2.imwrite("static/heatmap.jpg", overlay)

    # -----------------------------
    # DISPLAY OUTPUT
    # -----------------------------
    cv2.imshow("Retail AI Heatmap", overlay)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -----------------------------
# RELEASE RESOURCES
# -----------------------------
cap.release()
cv2.destroyAllWindows()