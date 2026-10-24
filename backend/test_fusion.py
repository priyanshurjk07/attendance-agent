import time
import json
import cv2
import sounddevice as sd
from core.camera_service import CameraService
from core.face_detection import FaceDetector, SimpleTracker
from core.audio_analysis import AudioAnalyzer
from core.fusion import SignalFusion

DURATION = 10

analyzer = AudioAnalyzer()
camera = CameraService()
detector = FaceDetector()
tracker = SimpleTracker()

camera.start()
print("Recording camera + microphone for 10 seconds...")
print("Plan: talk for the first 4 seconds, then stay quiet and still,")
print("then move your head around in the last 3 seconds.")

print("Get ready... starting in 3 seconds")
warm = time.time()
while time.time() - warm < 3:
    f = camera.get_frame()
    if f is not None:
        cv2.putText(f, "Get ready...", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Fusion Test", f)
    cv2.waitKey(1)
print("GO! Talk now")

audio = sd.rec(int(DURATION * analyzer.sample_rate),
               samplerate=analyzer.sample_rate, channels=1, dtype="float32")
start = time.time()
visual = []
prev_center = None

while time.time() - start < DURATION:
    frame = camera.get_frame()
    if frame is None:
        print("Could not read from camera")
        break

    t = time.time() - start
    detections = detector.process_frame(frame)
    tracked = tracker.update(detections)

    movement = 0.0
    if tracked:
        x, y, w, h = tracked[0]["bbox"]
        center = (x + w / 2, y + h / 2)
        if prev_center is not None:
            movement = ((center[0] - prev_center[0]) ** 2 +
                        (center[1] - prev_center[1]) ** 2) ** 0.5
        prev_center = center
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        prev_center = None

    visual.append({"time": round(t, 2), "faces": len(tracked),
                   "movement": round(movement, 1)})
    cv2.imshow("Fusion Test", frame)
    cv2.waitKey(1)

camera.stop()
cv2.destroyAllWindows()
sd.wait()

windows = analyzer.analyze(analyzer.extract(audio[:, 0]))["windows"]
fused = SignalFusion(interval_seconds=2).fuse(visual, windows)

print()
for s in fused:
    print(f"{s['start']:>4}-{s['end']:<4}s  face={s['face_present_ratio']}  "
          f"move={s['avg_movement']:>5}  speech={s['speech_ratio']}  "
          f"silence={s['silence_ratio']}  amp={s['avg_amplitude']}")

with open("fused_data.json", "w") as f:
    json.dump(fused, f, indent=2)
print("Saved fused_data.json")