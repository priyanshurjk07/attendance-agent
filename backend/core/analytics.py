import json
import os
from statistics import pstdev


class AnalyticsService:
    """Builds a reflective, privacy-friendly session summary for one participant."""

    def __init__(self, attendance_path="attendance_log.json"):
        self.attendance_path = attendance_path

    def _load_attendance(self):
        if os.path.exists(self.attendance_path):
            with open(self.attendance_path) as f:
                return json.load(f)
        return []

    def generate(self, result):
        segs = result["segments"]
        if not segs:
            return {}

        interval = segs[0]["end"] - segs[0]["start"]
        total_time = len(segs) * interval
        scores = [s["score"] for s in segs]

        # Attendance (from the vision module)
        log = self._load_attendance()
        attendance = {
            "marked_present": len(log) > 0,
            "marked_present_at": log[0]["marked_present_at"] if log else None,
            "session_seconds": total_time,
        }

        # Participation
        face_time = sum(s["face_present_ratio"] for s in segs) * interval
        speech_time = sum(s["speech_ratio"] for s in segs) * interval
        focused_time = sum(interval for s in segs if s["level"] in ("high", "medium"))

        # Consistency: how much the score moved around
        spread = pstdev(scores) if len(scores) > 1 else 0
        if spread < 10:
            consistency = "steady"
        elif spread < 20:
            consistency = "moderately varied"
        else:
            consistency = "highly varied"

        best = max(segs, key=lambda s: s["score"])
        lowest = min(segs, key=lambda s: s["score"])

        # Reflective notes (no grades, no comparison with others)
        notes = [f"Your engagement was {consistency} during this session."]
        notes.append(f"Your strongest focus was around {best['start']}-{best['end']}s.")
        if lowest["level"] == "low":
            notes.append(f"Attention dipped around {lowest['start']}-{lowest['end']}s. "
                         "Noticing when this happens can help you plan short breaks.")
        if speech_time == 0:
            notes.append("You did not speak in this session. Asking or answering one question can help you stay involved.")

        return {
            "attendance": attendance,
            "participation": {
                "visible_seconds": round(face_time, 1),
                "speaking_seconds": round(speech_time, 1),
                "focused_seconds": round(focused_time, 1),
                "average_score": result["average_score"],
                "overall_level": result["overall_level"],
                "consistency": consistency,
            },
            "timeline": [
                {"time": f"{s['start']}-{s['end']}s", "score": s["score"], "level": s["level"]}
                for s in segs
            ],
            "reflection": notes,
            "privacy_note": "This summary describes only your own session. It is not a grade and is not compared with other participants.",
        }