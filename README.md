## VIRTUAL SCROLLER 


```markdown
# Virtual Scroller

A camera-based, hands-free desktop scroller in Python using **OpenCV**, **MediaPipe Tasks Vision** (`FaceLandmarker` and `HandLandmarker`), and **PyAutoGUI** to scroll PDFs, web browsers, and code editors globally via hand pinch height and mouth movements.

## Overview & Learning Path

Designed as a hands-on computer vision and systems project that can be prototyped in **8–15 hours** if you know basic Python and virtual environments (or **3–5 weeks** starting from zero programming fundamentals).

1. **Stage 1 — Webcam Loop & Frame Basics (OpenCV):** Capture frames with `cv2.VideoCapture(0)`, mirror horizontally with `cv2.flip(frame, 1)`, overlay HUD text with `cv2.putText`, and exit cleanly on `q` (`cap.release()` and `cv2.destroyAllWindows()`).
2. **Stage 2 — Face & Hand Landmark Extraction (MediaPipe Tasks):** Convert OpenCV BGR frames to RGB (`mp.ImageFormat.SRGB`), pass monotonically increasing millisecond timestamps to `FaceLandmarker` and `HandLandmarker` in `VIDEO` mode, and read normalized `(x, y)` coordinates.
3. **Stage 3 — Scale-Invariant Ratios, Deadzones & OS Scroll Triggers:** Normalize mouth opening (`13` to `14`) and mouth width (`61` to `291`) by face height (`10` to `152`) so sitting closer or farther from the webcam does not distort readings, add deadzones, and trigger scroll events with `pyautogui.scroll()`.
4. **Stage 4 — Proportional Speed, Live Calibration & Persistence:** Scale scroll intensity from activation thresholds up to maximum ceilings, add a `c`-key calibration routine for neutral mouth ratios and resting hand deadzone center, and persist settings to `calibration.json`.

## Features

- **Proportional Hand Pinch Scrolling:** Pinch thumb tip (`4`) and index fingertip (`8`) below `PINCH_ENGAGE_THRESH` to engage hand scrolling (taking priority over mouth gestures). Moving above the top deadzone line scrolls **up**; moving below the bottom deadzone line scrolls **down**, with step size scaling smoothly from `1` up to `MAX_HAND_SCROLL_STEP` as distance from the deadzone increases.
- **Proportional Mouth Gestures:** Dropping your jaw (`mouth_open_ratio > mouth_open_down_thresh`) scrolls **down** proportionally up to `MOUTH_OPEN_MAX_RATIO` (`MAX_MOUTH_SCROLL_STEP`). Puckering your lips (`mouth_width_ratio < mouth_pucker_up_thresh`) scrolls **up** proportionally down to `MOUTH_PUCKER_MIN_RATIO`.
- **One-Key Calibration (`c`) + Auto-Persistence:** Pressing `c` in the preview window samples your relaxed mouth proportions and resting hand vertical position (`index_y`) over `1.0` second, updates `mouth_open_down_thresh`, `mouth_pucker_up_thresh`, and `hand_center_y` live, and saves them to `calibration.json` (created automatically; no manual JSON creation required).
- **Quick Toggle & Safe Exit:** Press `s` to pause or resume OS scrolling (`ON` / `PAUSED`) and `q` to quit cleanly.

## Project Structure

```text
Virtual Scroller/
├── models/                 # Auto-downloaded MediaPipe .task bundles (gitignored)
│   ├── face_landmarker.task
│   └── hand_landmarker.task
├── sandbox_vision.py       # Landmark & ratio inspection script (no scrolling)
├── scroller.py             # Main system-wide camera scroller with calibration
├── calibration.json        # Auto-generated local thresholds & hand deadzone (gitignored)
├── .gitignore
├── requirements.txt
└── README.md

```

## Setup & Installation

1. **Create and activate a virtual environment inside the project folder:**
```bash
cd "/Users/kirtan/Documents/EDU/CSE 1/DEVS/Virtual Scroller"
python3 -m venv venv
source venv/bin/activate

```


2. **Install dependencies:**
```bash
pip install opencv-python mediapipe pyautogui certifi

```


Or export and install from `requirements.txt`:
```bash
pip freeze > requirements.txt
pip install -r requirements.txt

```


3. **Grant macOS Accessibility permission (required for `pyautogui.scroll()`):**
* Open **System Settings** -> **Privacy & Security** -> **Accessibility**.
* Enable **Terminal** (or whichever terminal/IDE runs `scroller.py`).
* If it was already enabled before running `pyautogui`, toggle it off and back on, then restart your terminal or editor session.



## Usage

### 1. Inspect Live Ratios First (`sandbox_vision.py`)

Run `sandbox_vision.py` before enabling OS scrolling to observe your neutral and active `mouth_open`, `mouth_width`, `pinch_ratio`, and `index_y` values in the terminal and preview window:

```bash
python sandbox_vision.py

```

### 2. Run the Camera Scroller (`scroller.py`)

Start `scroller.py`, focus your target PDF viewer, browser tab, or editor window, and use gestures or hotkeys:

```bash
python scroller.py

```

#### Keyboard Controls (in the OpenCV preview window)

* **`c`**: Start a `1.0`-second calibration window—hold a relaxed neutral mouth and rest your hand where you want the neutral deadzone center (`hand_center_y`). Updates thresholds/deadzone live and saves to `calibration.json`.
* **`s`**: Pause or resume OS scroll calls (`ON` / `PAUSED`).
* **`q`**: Exit the loop, release the camera, and close OpenCV windows.

#### Tunable Constants (`scroller.py`)

* **Activation thresholds:** `MOUTH_OPEN_DOWN_THRESH`, `MOUTH_PUCKER_UP_THRESH`, `PINCH_ENGAGE_THRESH`, `HAND_CENTER_Y`, `HAND_DEADZONE_Y`
* **Speed ceilings & calibration margins:** `MOUTH_OPEN_MAX_RATIO`, `MOUTH_PUCKER_MIN_RATIO`, `CALIBRATION_DURATION_SEC`, `OPEN_MARGIN_RATIO`, `PUCKER_SCALE_FACTOR`
* **Scroll step & pacing:** `MAX_MOUTH_SCROLL_STEP`, `MAX_HAND_SCROLL_STEP`, `SCROLL_INTERVAL_SEC`

If `calibration.json` exists in the project directory on startup, `load_calibration()` loads those saved values automatically; otherwise it uses the defaults in `scroller.py`. Deleting `calibration.json` restores defaults on next launch.

## Troubleshooting Common Errors

* **`AttributeError: module 'mediapipe' has no attribute 'solutions'`:** Newer MediaPipe releases removed `mp.solutions` in favor of `mp.tasks.vision` (`FaceLandmarker` and `HandLandmarker` with `.task` model files). Either use the `mp.tasks.vision` implementation or install a legacy release that still provides `mp.solutions` (`pip install "mediapipe==0.10.14"`).
* **`ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED]` when downloading `.task` models on macOS:** Pass a `certifi` SSL context to `urllib.request.urlopen` (`ssl.create_default_context(cafile=certifi.where())`) inside `ensure_model()`, or download `face_landmarker.task` and `hand_landmarker.task` directly into `models/` with `curl -L`.
* **Neutral mouth triggers upward scrolling:** Your resting `mouth_width_ratio` is below `MOUTH_PUCKER_UP_THRESH`. Relax your face and press `c` to recalibrate automatically, or lower `MOUTH_PUCKER_UP_THRESH` in `scroller.py`.

## Uploading to GitHub via SSH

1. **Create `.gitignore` (do not paste raw `.gitignore` lines directly into `zsh` as commands):**
```bash
cat << 'EOF' > .gitignore
venv/
__pycache__/
*.py[cod]
models/
calibration.json
.DS_Store
EOF

```


2. **Generate an SSH key (if needed), add it to `ssh-agent` and macOS Keychain, and copy the public key:**
```bash
ls -al ~/.ssh
ssh-keygen -t ed25519 -C "your_email@example.com"
eval "$(ssh-agent -s)"
ssh-add --apple-use-keychain ~/.ssh/id_ed25519
pbcopy < ~/.ssh/id_ed25519.pub

```


3. **Add the key on GitHub** under **Settings** -> **SSH and GPG keys** -> **New SSH key**, then test the connection:
```bash
ssh -T git@github.com

```


4. **Initialize Git, commit, and push over SSH to an empty GitHub repository:**
```bash
cd "/Users/kirtan/Documents/EDU/CSE 1/DEVS/Virtual Scroller"
git init
git add .
git commit -m "Add camera-based hand and mouth scroller with calibration"
git branch -M main
git remote add origin git@github.com:<your-username>/virtual-scroller.git
git push -u origin main

```
