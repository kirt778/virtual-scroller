import json
import math
import ssl
import time
import urllib.request
from pathlib import Path

import certifi
import cv2
import mediapipe as mp
import pyautogui
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
CONFIG_PATH = BASE_DIR / "calibration.json"

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

# --- 1. DEFAULT ACTIVATION THRESHOLDS ---
MOUTH_OPEN_DOWN_THRESH = 0.065   # Jaw opening above this starts scrolling down
MOUTH_PUCKER_UP_THRESH = 0.210   # Mouth width below this starts scrolling up
PINCH_ENGAGE_THRESH = 0.250      # Pinch ratio below this engages hand scrolling
HAND_CENTER_Y = 0.50             # Default vertical center of deadzone (0.0 top .. 1.0 bottom)
HAND_DEADZONE_Y = 0.04#0.08           # Half-width of neutral band around hand_center_y

# --- 2. FULL-SPEED CEILINGS & CALIBRATION MARGINS ---
MOUTH_OPEN_MAX_RATIO = 0.220     # Wide open mouth ceiling for max down speed
MOUTH_PUCKER_MIN_RATIO = 0.140   # Tight pucker floor for max up speed
CALIBRATION_DURATION_SEC = 1.0   # Seconds to sample neutral face & resting hand on 'c'
OPEN_MARGIN_RATIO = 0.040        # Added above neutral open ratio for down threshold
PUCKER_SCALE_FACTOR = 0.85       # Multiplied by neutral width ratio for up threshold

# --- 3. SCROLL SPEED & TIMING ---
MAX_MOUTH_SCROLL_STEP = 10       # Max scroll lines per tick for mouth gestures
MAX_HAND_SCROLL_STEP = 14        # Max scroll lines per tick for hand gestures
SCROLL_INTERVAL_SEC = 0.05       # Minimum seconds between OS scroll calls


def ensure_model(url: str, dest: Path) -> str:
    """Download a MediaPipe .task bundle once using certifi if missing."""
    if not dest.exists():
        print(f"Downloading {dest.name}...")
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(url, context=ssl_context) as resp, open(dest, "wb") as out:
            out.write(resp.read())
    return str(dest)


def euclidean_distance(pt_a, pt_b) -> float:
    """Compute 2D Euclidean distance between two normalized landmarks."""
    return math.hypot(pt_a.x - pt_b.x, pt_a.y - pt_b.y)


def clamp(value: float, low: float, high: float) -> float:
    """Clamp a float into [low, high]."""
    return max(low, min(high, value))


def scale_intensity(value: float, start_thresh: float, full_thresh: float, max_step: int) -> int:
    """Map value from start_thresh -> full_thresh onto [1, max_step]."""
    span = full_thresh - start_thresh
    if abs(span) < 1e-9:
        return 1
    progress = clamp((value - start_thresh) / span, 0.0, 1.0)
    return max(1, int(round(progress * max_step)))


def draw_points(frame, landmarks, color=(0, 255, 0), radius=2):
    """Draw landmark dots on the video frame."""
    h, w, _ = frame.shape
    for lm in landmarks:
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), radius, color, -1)


def load_calibration() -> tuple[float, float, float]:
    """Load saved mouth thresholds and hand center from calibration.json if present."""
    open_thresh = MOUTH_OPEN_DOWN_THRESH
    pucker_thresh = MOUTH_PUCKER_UP_THRESH
    hand_center = HAND_CENTER_Y

    if CONFIG_PATH.exists():
        try:
            with CONFIG_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if "mouth_open_down_thresh" in data:
                open_thresh = float(data["mouth_open_down_thresh"])
            if "mouth_pucker_up_thresh" in data:
                pucker_thresh = float(data["mouth_pucker_up_thresh"])
            if "hand_center_y" in data:
                hand_center = clamp(float(data["hand_center_y"]), 0.20, 0.80)
            print(
                f"Loaded {CONFIG_PATH.name}: Up<{pucker_thresh:.3f}, "
                f"Dn>{open_thresh:.3f}, HandY={hand_center:.3f}"
            )
        except (OSError, KeyError, ValueError, TypeError) as exc:
            print(f"Could not load {CONFIG_PATH.name} ({exc}); using defaults.")

    return open_thresh, pucker_thresh, hand_center


def save_calibration(open_thresh: float, pucker_thresh: float, hand_center: float) -> None:
    """Persist calibrated mouth thresholds and hand center to calibration.json."""
    payload = {
        "mouth_open_down_thresh": round(open_thresh, 4),
        "mouth_pucker_up_thresh": round(pucker_thresh, 4),
        "hand_center_y": round(hand_center, 4),
    }
    try:
        with CONFIG_PATH.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved calibration to {CONFIG_PATH.name}.")
    except OSError as exc:
        print(f"Warning: could not save {CONFIG_PATH.name}: {exc}")


def compute_scroll_step(
    mouth_open_ratio,
    mouth_width_ratio,
    pinch_ratio,
    index_y,
    mouth_open_down_thresh: float,
    mouth_pucker_up_thresh: float,
    hand_center_y: float,
):
    """Return (signed_scroll_step, mode_label). Positive = up, negative = down."""
    # 1. Hand pinch clutch takes priority over mouth gestures.
    if pinch_ratio is not None and index_y is not None and pinch_ratio < PINCH_ENGAGE_THRESH:
        upper_edge = clamp(hand_center_y - HAND_DEADZONE_Y, 0.05, 0.90)
        lower_edge = clamp(hand_center_y + HAND_DEADZONE_Y, 0.10, 0.95)

        if index_y < upper_edge:
            step = scale_intensity(index_y, upper_edge, 0.05, MAX_HAND_SCROLL_STEP)
            return step, f"HAND UP (+{step})"

        if index_y > lower_edge:
            step = scale_intensity(index_y, lower_edge, 0.95, MAX_HAND_SCROLL_STEP)
            return -step, f"HAND DOWN (-{step})"

        return 0, "HAND PINCH -> DEADZONE"

    # 2. Mouth open -> proportional scroll DOWN.
    if mouth_open_ratio is not None and mouth_open_ratio > mouth_open_down_thresh:
        step = scale_intensity(
            mouth_open_ratio,
            mouth_open_down_thresh,
            MOUTH_OPEN_MAX_RATIO,
            MAX_MOUTH_SCROLL_STEP,
        )
        return -step, f"MOUTH OPEN DOWN (-{step})"

    # 3. Mouth pucker -> proportional scroll UP.
    if mouth_width_ratio is not None and mouth_width_ratio < mouth_pucker_up_thresh:
        step = scale_intensity(
            mouth_width_ratio,
            mouth_pucker_up_thresh,
            MOUTH_PUCKER_MIN_RATIO,
            MAX_MOUTH_SCROLL_STEP,
        )
        return step, f"MOUTH PUCKER UP (+{step})"

    return 0, "IDLE"


def finalize_calibration(
    mouth_samples: list[tuple[float, float]],
    hand_y_samples: list[float],
    current_open_thresh: float,
    current_pucker_thresh: float,
    current_hand_center: float,
) -> tuple[float, float, float]:
    """Update mouth thresholds and resting hand deadzone center from collected samples."""
    new_open_thresh = current_open_thresh
    new_pucker_thresh = current_pucker_thresh
    new_hand_center = current_hand_center

    if mouth_samples:
        avg_open = sum(s[0] for s in mouth_samples) / len(mouth_samples)
        avg_width = sum(s[1] for s in mouth_samples) / len(mouth_samples)
        new_open_thresh = max(0.035, avg_open + OPEN_MARGIN_RATIO)
        new_pucker_thresh = avg_width * PUCKER_SCALE_FACTOR
        print(
            f"Face ({len(mouth_samples)} frames): neutral_open={avg_open:.3f} -> Dn>{new_open_thresh:.3f}, "
            f"neutral_width={avg_width:.3f} -> Up<{new_pucker_thresh:.3f}"
        )
    else:
        print("Calibration warning: no face detected during window; kept existing mouth thresholds.")

    if hand_y_samples:
        avg_hand_y = sum(hand_y_samples) / len(hand_y_samples)
        new_hand_center = clamp(avg_hand_y, 0.20, 0.80)
        print(
            f"Hand ({len(hand_y_samples)} frames): resting_y={avg_hand_y:.3f} -> "
            f"deadzone=[{new_hand_center - HAND_DEADZONE_Y:.3f}, {new_hand_center + HAND_DEADZONE_Y:.3f}]"
        )
    else:
        print("Calibration note: no hand detected during window; kept existing hand deadzone.")

    save_calibration(new_open_thresh, new_pucker_thresh, new_hand_center)
    return new_open_thresh, new_pucker_thresh, new_hand_center


def main():
    face_model_file = ensure_model(FACE_MODEL_URL, FACE_MODEL_PATH)
    hand_model_file = ensure_model(HAND_MODEL_URL, HAND_MODEL_PATH)

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
        print("Error: Could not open webcam.")
        return

    mouth_open_down_thresh, mouth_pucker_up_thresh, hand_center_y = load_calibration()
    mouth_samples: list[tuple[float, float]] = []
    hand_y_samples: list[float] = []
    calibrating_until = 0.0

    scroll_enabled = True
    last_scroll_time = 0.0
    start_time = time.monotonic()

    print("Controls: 'c' = calibrate neutral mouth + resting hand deadzone (1s), 's' = pause/resume, 'q' = quit.")

    with (
        vision.FaceLandmarker.create_from_options(face_options) as face_landmarker,
        vision.HandLandmarker.create_from_options(hand_options) as hand_landmarker,
    ):
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((time.monotonic() - start_time) * 1000)

            face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
            hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

            mouth_open_ratio = None
            mouth_width_ratio = None
            pinch_ratio = None
            index_y = None

            if face_result.face_landmarks:
                face_lms = face_result.face_landmarks[0]
                draw_points(frame, face_lms, color=(200, 220, 0), radius=1)
                face_height = euclidean_distance(face_lms[10], face_lms[152])
                if face_height > 1e-5:
                    mouth_open_ratio = euclidean_distance(face_lms[13], face_lms[14]) / face_height
                    mouth_width_ratio = euclidean_distance(face_lms[61], face_lms[291]) / face_height

            if hand_result.hand_landmarks:
                hand_lms = hand_result.hand_landmarks[0]
                draw_points(frame, hand_lms, color=(0, 255, 120), radius=4)
                wrist = hand_lms[0]
                thumb_tip = hand_lms[4]
                index_tip = hand_lms[8]
                hand_scale = euclidean_distance(wrist, index_tip)
                pinch_ratio = euclidean_distance(thumb_tip, index_tip) / max(hand_scale, 1e-5)
                index_y = index_tip.y

            now = time.monotonic()

            if now < calibrating_until:
                if mouth_open_ratio is not None and mouth_width_ratio is not None:
                    mouth_samples.append((mouth_open_ratio, mouth_width_ratio))
                if index_y is not None:
                    hand_y_samples.append(index_y)
                mode_label = "CALIBRATING MOUTH + HAND ZONE..."
                step = 0
            else:
                if calibrating_until > 0.0:
                    mouth_open_down_thresh, mouth_pucker_up_thresh, hand_center_y = finalize_calibration(
                        mouth_samples,
                        hand_y_samples,
                        mouth_open_down_thresh,
                        mouth_pucker_up_thresh,
                        hand_center_y,
                    )
                    mouth_samples.clear()
                    hand_y_samples.clear()
                    calibrating_until = 0.0

                step, mode_label = compute_scroll_step(
                    mouth_open_ratio,
                    mouth_width_ratio,
                    pinch_ratio,
                    index_y,
                    mouth_open_down_thresh,
                    mouth_pucker_up_thresh,
                    hand_center_y,
                )

                if scroll_enabled and step != 0 and (now - last_scroll_time) >= SCROLL_INTERVAL_SEC:
                    pyautogui.scroll(step)
                    last_scroll_time = now

            top_deadzone_px = int(clamp(hand_center_y - HAND_DEADZONE_Y, 0.05, 0.90) * h)
            bot_deadzone_px = int(clamp(hand_center_y + HAND_DEADZONE_Y, 0.10, 0.95) * h)
            cv2.line(frame, (0, top_deadzone_px), (w, top_deadzone_px), (255, 200, 0), 1)
            cv2.line(frame, (0, bot_deadzone_px), (w, bot_deadzone_px), (255, 200, 0), 1)

            state_tag = "ON" if scroll_enabled else "PAUSED"
            cv2.putText(
                frame,
                f"[{state_tag}] {mode_label} | HandY:{hand_center_y:.2f} Up<{mouth_pucker_up_thresh:.3f} Dn>{mouth_open_down_thresh:.3f}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Vision Scroller (C: calibrate, S: toggle, Q: quit)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                scroll_enabled = not scroll_enabled
            elif key == ord("c") and calibrating_until == 0.0:
                mouth_samples.clear()
                hand_y_samples.clear()
                calibrating_until = now + CALIBRATION_DURATION_SEC
                print("Calibrating for 1.0s—hold a neutral mouth and rest your hand where you want the deadzone...")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()