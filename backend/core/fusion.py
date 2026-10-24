class SignalFusion:
    """Combines vision samples and audio windows into time segments."""

    def __init__(self, interval_seconds=2):
        self.interval = interval_seconds

    def fuse(self, visual_samples, audio_windows):
        times = [v["time"] for v in visual_samples] + [a["time"] for a in audio_windows]
        if not times:
            return []
        end_time = max(times)

        segments = []
        start = 0.0
        while start <= end_time:
            stop = start + self.interval
            v = [x for x in visual_samples if start <= x["time"] < stop]
            a = [x for x in audio_windows if start <= x["time"] < stop]

            face_ratio = sum(1 for x in v if x["faces"] > 0) / len(v) if v else 0.0
            movement = sum(x["movement"] for x in v) / len(v) if v else 0.0

            def ratio(label):
                return sum(1 for x in a if x["label"] == label) / len(a) if a else 0.0

            segments.append({
                "start": round(start, 1),
                "end": round(stop, 1),
                "face_present_ratio": round(face_ratio, 2),
                "avg_movement": round(movement, 1),
                "speech_ratio": round(ratio("speech"), 2),
                "noise_ratio": round(ratio("noise"), 2),
                "silence_ratio": round(ratio("silence"), 2),
                "avg_amplitude": round(sum(x["amplitude"] for x in a) / len(a), 4) if a else 0.0,
            })
            start = stop
        return segments