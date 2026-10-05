# Real-Time Squat Form & Kinematics Tracker

A lightweight, real-time computer vision pipeline designed to analyze human biomechanics during squat execution using single-camera video streams. The system computes joint kinematics, enforces full depth validation via a Finite State Machine (FSM), and filters real-world sensory noise.

## Key Features

- **MediaPipe Tasks API (Vision Running Mode):** Utilizes the modern MediaPipe 0.10+ Tasks architecture configured for standalone CPU execution, eliminating Apple Silicon Metal hardware runtime conflicts and latency spikes.
- **Vector-Based Kinematics:** Computes the interior angle of the knee joint in Euclidean 2D space using dot products and norms of vectors constructed from normalized anatomical landmarks.
- **Finite State Machine (FSM):** Robust state tracking (`UP` vs `DOWN`) with hysteresis thresholds to prevent edge-case state oscillation.
- **Signal Filtering & Noise Rejection:**
  - **Exponential Moving Average (EMA):** Dampens high-frequency frame-to-frame landmark jitter.
  - **Temporal Validation (Debounce):** Rejects invalid movements (e.g., walking, shifting) by requiring valid repetitions to occur within an athletic time window ($0.8\text{s} \le \Delta t \le 4.0\text{s}$).
- **Kinematic Depth Enforcement:** Repetitions are only credited if the minimum knee flexion reaches $\le 95^\circ$.

---

## How to Use & Camera Setup

1. **Camera Placement:**
   - Position your device approximately 2–3 meters away, roughly at hip-to-chest height.
   - **Crucial Field of View:** Ensure the camera captures your entire torso and legs at least **15–20 cm below the knee** (ideally including the ankle joint) across the full range of motion.
2. **Body Positioning:**
   - Stand in a **lateral (side-profile) orientation** relative to the camera lens. The 2D joint angle estimation is designed for profile kinematic tracking.
3. **10-Second Warm-up Countdown:**
   - Upon launching the script, a 10-second countdown (`GET READY`) will appear on the display.
   - Use this time to step back, center yourself in the frame, and assume the starting position. Repetition tracking is paused during this period.
4. **Repetition Execution:**
   - **Start:** Stand upright with knees extended (`STAGE: UP`).
   - **Descent:** Squat past parallel until the knee angle drops to $\le 95^\circ$ (`STAGE: DOWN`).
   - **Ascent:** Return to the starting position at a controlled pace ($0.8\text{s} - 4.0\text{s}$).
   - **Feedback:** The dashboard will increment `REPS` and display `GOOD REP!`. If the squat is too shallow, too quick, or paused for too long, clear visual feedback (`SQUAT DEEPER!`, `TOO FAST!`, `TOO SLOW!`) will be displayed.

---

## Kinematics Pipeline

1. **Spatial Representation:** The video frame is extracted as an array $(H, W, 3)$ and normalized into spatial coordinates $x, y \in [0.0, 1.0]$.
2. **Vector Construction:** Vectors are formed between anatomical landmarks:
   - $\vec{BA} = \text{Hip} - \text{Knee}$
   - $\vec{BC} = \text{Ankle} - \text{Knee}$
3. **Angle Calculation:**
   $$\theta = \arccos\left(\frac{\vec{BA} \cdot \vec{BC}}{\Vert{}\vec{BA}\Vert{} \Vert{}\vec{BC}\Vert{}}\right)$$
4. **Signal Smoothing:**
   $$\theta_{\text{smooth}} = 0.7 \cdot \theta_{\text{raw}} + 0.3 \cdot \theta_{\text{prev}}$$

---

## Tech Stack

- **Language:** Python 3.10+
- **Computer Vision:** OpenCV (`cv2`)
- **Pose Estimation:** Google MediaPipe (Tasks API, BlazePose backbone)
- **Math & Vector Operations:** NumPy

---

## Getting Started

### 1. Clone the repository
```bash
git clone [https://github.com/](https://github.com/)<your-username>/exerciseTrack.git
cd exerciseTrack
```

### 2. Set up environment & dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install opencv-python mediapipe numpy
```

### 3. Download the model asset
The Tasks API requires the pre-trained model asset bundle:
```bash
curl -L -o pose_landmarker.task [https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task)
```

### 4. Run the tracker
```bash
python3 track.py
```
*Press `q` on your keyboard to exit the stream.*

---

## Project Structure

```text
├── track.py                 # Core tracking script (Tasks API inference + FSM)
├── pose_landmarker.task     # MediaPipe pose weights bundle (git-ignored)
├── .gitignore               # Excludes binary assets, bytecode, and venv
└── README.md                # Technical documentation
```