import time
import cv2
import mediapipe as mp
import numpy as np

# Tasks API
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def calculate_angle(a, b, c):
    """Рахує кут у точці b (вершина) між векторами ba та bc."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-7)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# Стан трекера
counter = 0
stage = "UP"
feedback = "START"
min_angle = 180.0
smoothed_angle = None
start_down_time = None

# Налаштування таймера підготовки
COUNTDOWN_SECONDS = 10
start_session_time = time.time()

options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="pose_landmarker.task",
        delegate=BaseOptions.Delegate.CPU
    ),
    running_mode=VisionRunningMode.IMAGE
)

cap = cv2.VideoCapture(0)

with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = landmarker.detect(mp_image)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            
            # 23 - HIP, 25 - KNEE, 27 - ANKLE
            hip = [landmarks[23].x, landmarks[23].y]
            knee = [landmarks[25].x, landmarks[25].y]
            ankle = [landmarks[27].x, landmarks[27].y]
            
            raw_angle = calculate_angle(hip, knee, ankle)

            if smoothed_angle is None:
                smoothed_angle = raw_angle
            else:
                smoothed_angle = 0.7 * raw_angle + 0.3 * smoothed_angle

            # Отрисовка скелета
            hip_px = (int(hip[0] * w), int(hip[1] * h))
            knee_px = (int(knee[0] * w), int(knee[1] * h))
            ankle_px = (int(ankle[0] * w), int(ankle[1] * h))

            cv2.line(frame, hip_px, knee_px, (255, 255, 255), 3)
            cv2.line(frame, knee_px, ankle_px, (255, 255, 255), 3)
            cv2.circle(frame, hip_px, 6, (0, 0, 255), -1)
            cv2.circle(frame, knee_px, 6, (0, 255, 0), -1)
            cv2.circle(frame, ankle_px, 6, (0, 0, 255), -1)

            cv2.putText(frame, f"{int(smoothed_angle)} deg", (knee_px[0] - 20, knee_px[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        current_time = time.time()
        elapsed_time = current_time - start_session_time

        # Перевірка каунтдауну
        if elapsed_time < COUNTDOWN_SECONDS:
            remaining = int(COUNTDOWN_SECONDS - elapsed_time) + 1
            cv2.putText(frame, f"GET READY: {remaining}", (w // 2 - 180, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 4, cv2.LINE_AA)
            cv2.putText(frame, "Step into side-profile view", (w // 2 - 190, h // 2 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            # Основна FSM-логіка
            if smoothed_angle is not None:
                if smoothed_angle > 150:
                    if stage == "DOWN":
                        duration = current_time - start_down_time if start_down_time else 0
                        
                        if min_angle <= 95:
                            if 0.8 <= duration <= 4.0:
                                counter += 1
                                feedback = f"GOOD REP! ({duration:.1f}s)"
                            elif duration < 0.8:
                                feedback = "TOO FAST!"
                            else:
                                feedback = "TOO SLOW!"
                        else:
                            feedback = "SQUAT DEEPER!"

                        min_angle = 180.0
                        start_down_time = None

                    stage = "UP"

                elif smoothed_angle < 115:
                    if stage == "UP":
                        start_down_time = current_time
                    
                    stage = "DOWN"
                    if smoothed_angle < min_angle:
                        min_angle = smoothed_angle

        # Dashboard
        cv2.rectangle(frame, (0, 0), (640, 75), (30, 30, 30), -1)

        cv2.putText(frame, "REPS", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, str(counter), (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(frame, "STAGE", (110, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, stage, (110, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        color = (0, 255, 0) if "GOOD REP" in feedback else (0, 0, 255)
        if feedback == "START":
            color = (200, 200, 200)
        cv2.putText(frame, "FEEDBACK", (240, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, feedback, (240, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        cv2.imshow('Squat Form Tracker', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()