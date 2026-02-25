import cv2
import csv
import time
import mediapipe as mp
import numpy as np
from datetime import datetime


# ==========================
# MediaPipe Setup
# ==========================

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)


# ==========================
# Landmark Indexes
# ==========================

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

MOUTH = [13, 14]
LEFT_BROW = [70, 105]
RIGHT_BROW = [300, 334]


# ==========================
# Helper Functions
# ==========================

def dist(a, b):
    return np.linalg.norm(a - b)


def eye_aspect_ratio(eye):
    A = dist(eye[1], eye[5])
    B = dist(eye[2], eye[4])
    C = dist(eye[0], eye[3])

    return (A + B) / (2.0 * C)


# ==========================
# Gesture Detection
# ==========================

def detect_face_gesture(frame):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return "no_face"

    h, w, _ = frame.shape
    lm = result.multi_face_landmarks[0].landmark

    def get(idx):
        return np.array([lm[idx].x * w, lm[idx].y * h])


    # ----- Eyes (Blink) -----

    left_eye = np.array([get(i) for i in LEFT_EYE])
    right_eye = np.array([get(i) for i in RIGHT_EYE])

    ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2

    blink = ear < 0.20


    # ----- Mouth (Open / Smile) -----

    mouth_open = dist(get(13), get(14)) > 12

    smile_width = dist(get(61), get(291))
    face_width = dist(get(234), get(454))

    smile = smile_width / face_width > 0.45


    # ----- Brows (Surprise) -----

    brow_dist = (
        dist(get(LEFT_BROW[0]), get(LEFT_BROW[1])) +
        dist(get(RIGHT_BROW[0]), get(RIGHT_BROW[1]))
    ) / 2

    brow_raise = brow_dist > 22


    # ----- Head Nod (Simple) -----

    nose_y = get(1)[1]

    detect_face_gesture.prev_nose = getattr(
        detect_face_gesture, "prev_nose", nose_y
    )

    nod = abs(nose_y - detect_face_gesture.prev_nose) > 8
    detect_face_gesture.prev_nose = nose_y


    # ==========================
    # Classification
    # ==========================

    if blink:
        return "blink"

    if mouth_open and brow_raise:
        return "surprise"

    if smile:
        return "smile"

    if mouth_open:
        return "mouth_open"

    if nod:
        return "nod"

    return "neutral"


# ==========================
# Main Program
# ==========================

VIDEO_SOURCE = 1
OUTPUT_CSV = "face_gestures.csv"
FPS_TARGET = 3


cap = cv2.VideoCapture(VIDEO_SOURCE)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30

frame_interval = int(fps / FPS_TARGET)


with open(OUTPUT_CSV, "w", newline="") as csvfile:

    writer = csv.writer(csvfile)
    writer.writerow(["timestamp", "frame", "gesture"])

    frame_count = 0


    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break


        if frame_interval > 0 and frame_count % frame_interval == 0:

            gesture = detect_face_gesture(frame)

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

            writer.writerow([ts, frame_count, gesture])

            print(f"{frame_count:5d} | {gesture:10s} | {ts}")


        frame_count += 1


        cv2.imshow("Face Gesture Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


cap.release()
cv2.destroyAllWindows()

print("Saved to", OUTPUT_CSV)