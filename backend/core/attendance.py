import time
import json
from datetime import datetime


class AttendanceManager:
    def __init__(self, threshold_seconds=5):
        self.threshold = threshold_seconds
        self.presence = {}    # id -> total seconds visible
        self.last_seen = {}   # id -> last time seen
        self.present = set()  # ids marked present
        self.log = []

    def update(self, tracked_faces):
        now = time.time()
        for t in tracked_faces:
            tid = t["id"]
            # short gaps (under 2s) still count, so brief misses don't reset
            if tid in self.last_seen and now - self.last_seen[tid] < 2:
                self.presence[tid] = self.presence.get(tid, 0) + (now - self.last_seen[tid])
            else:
                self.presence.setdefault(tid, 0)
            self.last_seen[tid] = now

            if self.presence[tid] >= self.threshold and tid not in self.present:
                self.present.add(tid)
                self.log.append({
                    "participant_id": tid,
                    "presence_seconds": round(self.presence[tid], 1),
                    "marked_present_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })

    def save(self, path="attendance_log.json"):
        with open(path, "w") as f:
            json.dump(self.log, f, indent=2)