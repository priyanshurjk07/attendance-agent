import cv2


class FaceDetector:
    def __init__(self):
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(path)
        if self.face_cascade.empty():
            print("Error: Could not load face cascade xml")

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=6, minSize=(80, 80)
        )
        detections = []
        for (x, y, w, h) in faces:
            detections.append({"bbox": (int(x), int(y), int(w), int(h))})
        return detections



class SimpleTracker:
    """Gives each face an ID and keeps the same ID across frames."""

    def __init__(self, max_distance=250, max_missing=300):
        self.max_distance = max_distance
        self.max_missing = max_missing
        self.tracks = {}
        self.next_id = 1

    def update(self, detections):
        used = set()
        results = []
        for det in detections:
            x, y, w, h = det["bbox"]
            center = (x + w // 2, y + h // 2)

            best_id = None
            best_dist = self.max_distance
            for tid, t in self.tracks.items():
                if tid in used:
                    continue
                dist = ((center[0] - t["center"][0]) ** 2 +
                        (center[1] - t["center"][1]) ** 2) ** 0.5
                if dist < best_dist:
                    best_id, best_dist = tid, dist

            if best_id is None:
                best_id = self.next_id
                self.next_id += 1

            self.tracks[best_id] = {"center": center, "missing": 0}
            used.add(best_id)
            results.append({"id": best_id, "bbox": det["bbox"]})

        for tid in list(self.tracks):
            if tid not in used:
                self.tracks[tid]["missing"] += 1
                if self.tracks[tid]["missing"] > self.max_missing:
                    del self.tracks[tid]
        return results

