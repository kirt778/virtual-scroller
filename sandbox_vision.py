import math
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

# --- 1. MODEL DOWNLOAD & INITIALIZATION ---
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

FACE_MODEL_PATH = MODELS_DIR / "face_landmarker.task"
HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"

FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


import ssl
import certifi


def ensure_model(url: str, dest: Path) -> str:
    """Download a MediaPipe .task bundle once if not present locally."""
    if not dest.exists():
        print(f"Downloading {dest.name}...")
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(url, context=ssl_context) as response, open(dest, "wb") as out_file:
            out_file.write(response.read())
    return str(dest)


def euclidean_distance(pt_a, pt_b) -> float:
    """Compute 2D distance between two normalized landmarks (x, y in [0.0, 1.0])."""
    return math.hypot(pt_a.x - pt_b.x, pt_a.y - pt_b.y)


def draw_points(frame, landmarks, color=(0, 255, 0), radius=2):
    """Draw landmark dots on the OpenCV BGR frame."""
    h, w, _ = frame.shape
    for lm in landmarks:
        cx = int(lm.x * w)
        cy = int(lm.y * h)
        cv2.circle(frame, (cx, cy), radius, color, -1)


def main():
    face_model_file = ensure_model(FACE_MODEL_URL, FACE_MODEL_PATH)
    hand_model_file = ensure_model(HAND_MODEL_URL, HAND_MODEL_PATH)

    # VIDEO mode expects monotonically increasing timestamps in milliseconds.
    face_options = vision.FaceLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=face_model_file),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.6,
        min_face_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    hand_options = vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=hand_model_file),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam. Check camera permissions.")
        return

    last_print_time = 0.0
    print_interval = 0.25
    start_time = time.monotonic()

    print("Starting vision sandbox (Tasks API). Press 'q' in the video window to quit.")

    with (
        vision.FaceLandmarker.create_from_options(face_options) as face_landmarker,
        vision.HandLandmarker.create_from_options(hand_options) as hand_landmarker,
    ):
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Warning: Failed to read frame from camera.")
                break

            # Mirror horizontally so left/right matches what you see in the camera.
            frame = cv2.flip(frame, 1)

            # Convert BGR (OpenCV) -> RGB -> mp.Image for Tasks API.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((time.monotonic() - start_time) * 1000)

            face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
            hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

            mouth_open_ratio = None
            mouth_width_ratio = None
            pinch_ratio = None
            index_y = None

            # --- 2. PROCESS FACE LANDMARKS ---
            if face_result.face_landmarks:
                face_lms = face_result.face_landmarks[0]
                draw_points(frame, face_lms, color=(200, 220, 0), radius=1)

                # 10 = top of forehead, 152 = chin
                # 13 = inner upper lip, 14 = inner lower lip
                # 61 = left mouth corner, 291 = right mouth corner
                face_height = euclidean_distance(face_lms[10], face_lms[152])
                if face_height > 1e-5:
                    lip_gap = euclidean_distance(face_lms[13], face_lms[14])
                    mouth_span = euclidean_distance(face_lms[61], face_lms[291])

                    mouth_open_ratio = lip_gap / face_height
                    mouth_width_ratio = mouth_span / face_height

            # --- 3. PROCESS HAND LANDMARKS ---
            if hand_result.hand_landmarks:
                hand_lms = hand_result.hand_landmarks[0]
                draw_points(frame, hand_lms, color=(0, 255, 120), radius=4)

                # 0 = wrist, 4 = thumb tip, 8 = index fingertip
                wrist = hand_lms[0]
                thumb_tip = hand_lms[4]
                index_tip = hand_lms[8]

                hand_scale = euclidean_distance(wrist, index_tip)
                pinch_dist = euclidean_distance(thumb_tip, index_tip)

                pinch_ratio = pinch_dist / max(hand_scale, 1e-5)
                index_y = index_tip.y

            # --- 4. PRINT LIVE METRICS TO TERMINAL ---
            now = time.time()
            if now - last_print_time >= print_interval:
                last_print_time = now

                m_open = f"{mouth_open_ratio:.3f}" if mouth_open_ratio is not None else "  --- "
                m_width = f"{mouth_width_ratio:.3f}" if mouth_width_ratio is not None else "  --- "
                p_ratio = f"{pinch_ratio:.3f}" if pinch_ratio is not None else "  --- "
                idx_y = f"{index_y:.3f}" if index_y is not None else "  --- "

                print(
                    f"mouth_open={m_open} | "
                    f"mouth_width={m_width} | "
                    f"pinch_ratio={p_ratio} | "
                    f"index_y={idx_y}"
                )

            # --- 5. HUD & EXIT ---
            hud_text = (
                f"Open: {mouth_open_ratio:.3f}  Width: {mouth_width_ratio:.3f}"
                if mouth_open_ratio is not None and mouth_width_ratio is not None
                else "Face: not detected"
            )
            cv2.putText(
                frame,
                hud_text,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Vision Sandbox (Press Q to quit)", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()