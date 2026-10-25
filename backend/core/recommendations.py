class RecommendationEngine:
    """Rule-based teaching suggestions from engagement segments."""

    def __init__(self, drop_threshold=25, noise_limit=0.5,
                 face_limit=0.3, peak_speech=0.2):
        self.drop_threshold = drop_threshold
        self.noise_limit = noise_limit
        self.face_limit = face_limit
        self.peak_speech = peak_speech

    def _groups(self, indices):
        """Group consecutive segment numbers: [1,2,4] -> [[1,2],[4]]"""
        groups = []
        for i in indices:
            if groups and i == groups[-1][-1] + 1:
                groups[-1].append(i)
            else:
                groups.append([i])
        return groups

    def _span(self, segs, group):
        return f"{segs[group[0]]['start']}-{segs[group[-1]]['end']}s"

    def generate(self, result):
        segs = result["segments"]
        recs = []

        # Rule 1: low engagement
        low = [i for i, s in enumerate(segs) if s["level"] == "low"]
        for g in self._groups(low):
            span = self._span(segs, g)
            if len(g) >= 2:
                recs.append({"type": "alert", "time": span,
                             "message": "Extended low engagement. Try an interactive activity, a poll or a short break."})
            else:
                recs.append({"type": "warning", "time": span,
                             "message": "Brief low-engagement dip. A quick question can bring attention back."})

        # Rule 2: sudden attention drop
        for i in range(1, len(segs)):
            drop = segs[i - 1]["score"] - segs[i]["score"]
            if drop >= self.drop_threshold:
                recs.append({"type": "warning", "time": f"{segs[i]['start']}-{segs[i]['end']}s",
                             "message": f"Engagement fell by {round(drop)} points. Consider changing the pace or topic."})

        # Rule 3: noisy classroom
        noisy = [i for i, s in enumerate(segs) if s["noise_ratio"] >= self.noise_limit]
        for g in self._groups(noisy):
            recs.append({"type": "warning", "time": self._span(segs, g),
                         "message": "High background noise. Check classroom management or audio setup."})

        # Rule 4: participants not visible
        away = [i for i, s in enumerate(segs) if s["face_present_ratio"] < self.face_limit]
        for g in self._groups(away):
            recs.append({"type": "warning", "time": self._span(segs, g),
                         "message": "Participants were mostly out of camera view. Check camera angle or ask them to face the screen."})

        # Rule 5: peak interaction
        peak = [i for i, s in enumerate(segs)
                if s["level"] == "high" and s["speech_ratio"] >= self.peak_speech]
        for g in self._groups(peak):
            recs.append({"type": "info", "time": self._span(segs, g),
                         "message": "Peak interaction. This teaching style works well, so reuse it."})

        if not recs:
            recs.append({"type": "info", "time": "whole session",
                         "message": "No issues found. Maintain the current approach."})
        return recs