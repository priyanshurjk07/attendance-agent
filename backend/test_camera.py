import cv2
from core.camera_service import CameraService
from core.face_detection import FaceDetector

camera = CameraService()
detector = FaceDetector()
camera.start()
print("Press Q on the video window to quit")

while True:
    frame = camera.get_frame()
    if frame is None:
        print("Could not read from camera")
        break

    detections = detector.process_frame(frame)

    for d in detections:
        x, y, w, h = d["bbox"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.stop()
cv2.destroyAllWindows()