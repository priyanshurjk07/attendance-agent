class EngagementScorer:
    """Rule-based engagement score (0-100) for each time segment."""

    def __init__(self, weights=None):
        # Easy to tune: how much each signal counts (adds up to 100)
        self.weights = weights or {"presence": 40, "speech": 30, "movement": 20, "noise": 10}

    def score_segment(self, seg):
        w = self.weights

        # 1. Presence: face visible?
        presence = seg["face_present_ratio"] * w["presence"]

        # 2. Speech: someone talking?
        speech = min(seg["speech_ratio"] * 2, 1.0) * w["speech"]

        # 3. Movement: small natural movement is good, none or too much is not
        m = seg["avg_movement"]
        if 3 <= m <= 40:
            movement = w["movement"]
        elif m < 3:
            movement = w["movement"] * 0.5   # very still
        else:
            movement = w["movement"] * 0.3   # too much movement
        if seg["face_present_ratio"] == 0:
            movement = 0

        # 4. Noise: a quiet-ish room is better than a noisy one
        noise = (1 - seg["noise_ratio"]) * w["noise"]

        return round(presence + speech + movement + noise, 1)

    def level(self, score):
        if score >= 70:
            return "high"
        if score >= 40:
            return "medium"
        return "low"

    def score_session(self, segments):
        results = []
        for seg in segments:
            s = self.score_segment(seg)
            results.append({**seg, "score": s, "level": self.level(s)})
        avg = round(sum(r["score"] for r in results) / len(results), 1) if results else 0
        return {
            "segments": results,
            "average_score": avg,
            "overall_level": self.level(avg),
        }