from ultralytics import YOLO
import cv2
import sqlite3
import numpy as np
import time
from tracker import tracker
from flask import Flask, render_template, Response

app = Flask(__name__)

# =========================
# LOAD YOLO MODEL
# =========================

model = YOLO("yolov8n.pt")

# =========================
# VIDEO SOURCE SETTINGS
# =========================

USE_WEBCAM = True

VIDEO_PATH = "mall.mp4"

if USE_WEBCAM:
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(VIDEO_PATH)

# =========================
# COUNTS
# =========================

entry_count = 0
exit_count = 0

# Counting line
line_y = 250

# Tracking memory
person_positions = {}
counted_ids = set()

# Dwell time storage
entry_times = {}

def generate_frames():

    global entry_count, exit_count

    # Heatmap canvas
    heatmap = None

    while True:

        success, frame = cap.read()

        # =========================
        # VIDEO LOOP FIX
        # =========================

        if not success:

            # Restart video automatically
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            continue

        # Resize frame
        frame = cv2.resize(frame, (1280, 720))

        # Create heatmap canvas
        if heatmap is None:

            heatmap = np.zeros(
                (frame.shape[0], frame.shape[1]),
                dtype=np.float32
            )

        # =========================
        # RUN YOLO DETECTION
        # =========================

        results = model(frame)

        people_count = 0
        crowd_limit = 3

        detections = []

        for result in results:

            boxes = result.boxes

            for box in boxes:

                cls = int(box.cls[0])

                # Person class only
                if cls == 0:

                    people_count += 1

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    conf = float(box.conf[0])

                    detections.append(
                        ([x1, y1, x2 - x1, y2 - y1], conf, 'person')
                    )

        # =========================
        # DEEPSORT TRACKING
        # =========================

        tracks = tracker.update_tracks(
            detections,
            frame=frame
        )

        # Draw counting line
        cv2.line(
            frame,
            (0, line_y),
            (frame.shape[1], line_y),
            (0, 0, 255),
            3
        )

        for track in tracks:

            if not track.is_confirmed():
                continue

            track_id = track.track_id

            # Start dwell timer
            if track_id not in entry_times:

                entry_times[track_id] = time.time()

            # Dwell time
            dwell_time = int(
                time.time() - entry_times[track_id]
            )

            ltrb = track.to_ltrb()

            x1, y1, x2, y2 = map(int, ltrb)

            center_y = (y1 + y2) // 2
            center_x = (x1 + x2) // 2

            # =========================
            # HEATMAP
            # =========================

            cv2.circle(
                heatmap,
                (center_x, center_y),
                35,
                1,
                -1
            )

            # =========================
            # ENTRY / EXIT LOGIC
            # =========================

            if track_id in person_positions:

                previous_y = person_positions[track_id]

                # Entered
                if previous_y < line_y and center_y >= line_y:

                    if track_id not in counted_ids:

                        entry_count += 1
                        counted_ids.add(track_id)

                # Exited
                elif previous_y > line_y and center_y <= line_y:

                    if track_id not in counted_ids:

                        exit_count += 1
                        counted_ids.add(track_id)

            person_positions[track_id] = center_y

            # =========================
            # PRIVACY BLUR
            # =========================

            person_roi = frame[y1:y2, x1:x2]

            if person_roi.size > 0:

                blurred = cv2.GaussianBlur(
                    person_roi,
                    (99, 99),
                    30
                )

                frame[y1:y2, x1:x2] = blurred

            # =========================
            # DRAW BOX
            # =========================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # ID + dwell time
            cv2.putText(
                frame,
                f'ID: {track_id} | {dwell_time}s',
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        inside_store = entry_count - exit_count

        # =========================
        # DATABASE SAVE
        # =========================

        conn = sqlite3.connect("retail_ai.db")

        cursor = conn.cursor()

        cursor.execute("""

        INSERT INTO analytics(
            people_count,
            entry_count,
            exit_count,
            inside_count
        )

        VALUES (?, ?, ?, ?)

        """, (

            people_count,
            entry_count,
            exit_count,
            inside_store

        ))

        conn.commit()
        conn.close()

        # =========================
        # TEXT OVERLAY
        # =========================

        cv2.putText(
            frame,
            f'People Count: {people_count}',
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,255,255),
            2
        )

        cv2.putText(
            frame,
            f'Entered: {entry_count}',
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0,255,0),
            2
        )

        cv2.putText(
            frame,
            f'Exited: {exit_count}',
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0,0,255),
            2
        )

        cv2.putText(
            frame,
            f'Inside: {inside_store}',
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255,255,0),
            2
        )

        # =========================
        # CROWD ALERT
        # =========================

        if people_count >= crowd_limit:

            cv2.putText(
                frame,
                "CROWD ALERT!",
                (350, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0,0,255),
                3
            )

        # =========================
        # HEATMAP OVERLAY
        # =========================

        heatmap_blur = cv2.GaussianBlur(
            heatmap,
            (51,51),
            0
        )

        heatmap_norm = cv2.normalize(
            heatmap_blur,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        heatmap_uint8 = heatmap_norm.astype(np.uint8)

        heatmap_color = cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET
        )

        frame = cv2.addWeighted(
            frame,
            0.7,
            heatmap_color,
            0.3,
            0
        )

        # =========================
        # ENCODE FRAME
        # =========================

        ret, buffer = cv2.imencode('.jpg', frame)

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame +
            b'\r\n'
        )

# =========================
# DASHBOARD ROUTE
# =========================

@app.route('/')

def dashboard():

    conn = sqlite3.connect("retail_ai.db")

    cursor = conn.cursor()

    cursor.execute("""

    SELECT
        people_count,
        entry_count,
        exit_count,
        inside_count

    FROM analytics

    ORDER BY id DESC

    LIMIT 1

    """)

    data = cursor.fetchone()

    if data:

        people_count = data[0]
        entry_count = data[1]
        exit_count = data[2]
        inside_count = data[3]

    else:

        people_count = 0
        entry_count = 0
        exit_count = 0
        inside_count = 0

    # Total database entries
    cursor.execute(
        "SELECT COUNT(*) FROM analytics"
    )

    total_entries = cursor.fetchone()[0]

    # Chart data
    cursor.execute("""

    SELECT people_count
    FROM analytics
    ORDER BY id DESC
    LIMIT 10

    """)

    chart_data = cursor.fetchall()

    chart_data = [row[0] for row in chart_data]

    chart_data.reverse()

    conn.close()

    return render_template(

        "index.html",

        people_count=people_count,
        entry_count=entry_count,
        exit_count=exit_count,
        inside_count=inside_count,
        total_entries=total_entries,
        chart_data=chart_data

    )

# =========================
# VIDEO FEED ROUTE
# =========================

@app.route('/video_feed')

def video_feed():

    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# =========================
# RUN FLASK
# =========================

if __name__ == "__main__":
    app.run(debug=False)