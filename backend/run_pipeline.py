import sys
import time
import json
import numpy as np
import cv2
import sounddevice as sd

from core.camera_service import CameraService
from core.face_detection import FaceDetector, SimpleTracker
from core.attendance import AttendanceManager
from core.audio_analysis import AudioAnalyzer
from core.fusion import SignalFusion
from core.scoring import EngagementScorer
from core.recommendations import RecommendationEngine
from core.llm_insights import InsightGenerator
from core.analytics import AnalyticsService

DURATION = int(sys.argv[1]) if len(sys.argv) > 1 else 15

analyzer = AudioAnalyzer()
camera = CameraService()
detector = FaceDetector()
tracker = SimpleTracker()
attendance = AttendanceManager(threshold_seconds=5)

camera.start()

# 3-second countdown so the camera can adjust to the light
print("Get ready... starting in 3 seconds")
warm = time.time()
while time.time() - warm < 3:
    f = camera.get_frame()
    if f is not None:
        cv2.putText(f, "Get ready...", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Attendance & Engagement Agent", f)
    cv2.waitKey(1)

print(f"GO! Recording for {DURATION} seconds...")

# Start the microphone (if it fails, continue with video only)
audio = None
try:
    audio = sd.rec(int(DURATION * analyzer.sample_rate),
                   samplerate=analyzer.sample_rate, channels=1, dtype="float32")
except Exception as e:
    print(f"Microphone problem, continuing without audio: {e}")

start = time.time()
visual = []
prev_center = None

while time.time() - start < DURATION:
    frame = camera.get_frame()
    if frame is None:
        print("Could not read from camera")
        break

    t = time.time() - start
    tracked = tracker.update(detector.process_frame(frame))
    attendance.update(tracked)

    movement = 0.0
    if tracked:
        x, y, w, h = tracked[0]["bbox"]
        center = (x + w / 2, y + h / 2)
        if prev_center is not None:
            movement = ((center[0] - prev_center[0]) ** 2 +
                        (center[1] - prev_center[1]) ** 2) ** 0.5
        prev_center = center
    else:
        prev_center = None

    for tr in tracked:
        x, y, w, h = tr["bbox"]
        present = tr["id"] in attendance.present
        color = (0, 255, 0) if present else (0, 165, 255)
        label = "PRESENT" if present else "detecting..."
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"ID {tr['id']} {label}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.putText(frame, f"{DURATION - t:.0f}s left", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow("Attendance & Engagement Agent", frame)
    cv2.waitKey(1)

    visual.append({"time": round(t, 2), "faces": len(tracked),
                   "movement": round(movement, 1)})

camera.stop()
cv2.destroyAllWindows()

if audio is not None:
    sd.wait()
    y_audio = audio[:, 0]
else:
    y_audio = np.zeros(int(DURATION * analyzer.sample_rate), dtype="float32")

attendance.save("attendance_log.json")

# Run every stage of the pipeline
windows = analyzer.analyze(analyzer.extract(y_audio))["windows"]
fused = SignalFusion(interval_seconds=2).fuse(visual, windows)
result = EngagementScorer().score_session(fused)
recs = RecommendationEngine().generate(result)
insight = InsightGenerator().generate(result)
analytics = AnalyticsService().generate(result)

report = {
    "session_seconds": DURATION,
    "engagement": result,
    "recommendations": recs,
    "ai_insight": insight,
    "analytics": analytics,
}
with open("session_report.json", "w") as f:
    json.dump(report, f, indent=2)

# Print a readable summary
print()
print("=" * 50)
print("SESSION REPORT")
print("=" * 50)
print("Attendance:", "PRESENT" if analytics["attendance"]["marked_present"] else "not marked present")
print(f"Average engagement: {result['average_score']} ({result['overall_level']})")
print()
for s in result["segments"]:
    print(f"  {s['start']:>4}-{s['end']:<4}s  {s['score']:>5}  {s['level']}")
print()
print("Suggestions:")
for r in recs:
    print(f"  [{r['type'].upper()}] {r['time']}: {r['message']}")
print()
print(f"AI insight ({insight['source']}): {insight['insight']}")
print(f"AI recommendation: {insight['recommendation']}")
print()
print("Saved session_report.json")