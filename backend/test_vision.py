import cv2
from core.camera_service import CameraService
from core.face_detection import FaceDetector, SimpleTracker
from core.attendance import AttendanceManager

camera = CameraService()
detector = FaceDetector()
tracker = SimpleTracker()
attendance = AttendanceManager(threshold_seconds=5)

camera.start()
print("Camera started. Press Q on the video window to quit.")

while True:
    frame = camera.get_frame()
    if frame is None:
        print("Could not read from camera")
        break

    detections = detector.process_frame(frame)
    tracked = tracker.update(detections)
    attendance.update(tracked)

    for t in tracked:
        x, y, w, h = t["bbox"]
        tid = t["id"]
        secs = attendance.presence.get(tid, 0)
        is_present = tid in attendance.present
        status = "PRESENT" if is_present else "detecting..."
        color = (0, 255, 0) if is_present else (0, 165, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"ID {tid} | {secs:.1f}s | {status}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Attendance Agent", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.stop()
cv2.destroyAllWindows()
attendance.save()
print("Saved attendance_log.json")