import numpy as np
import librosa
import sounddevice as sd


class AudioAnalyzer:
    def __init__(self, sample_rate=16000, window_seconds=0.5):
        self.sample_rate = sample_rate
        self.window_seconds = window_seconds
        self.window_size = int(sample_rate * window_seconds)

    def record(self, seconds=5):
        """Record from the microphone (mono, 16000 Hz)."""
        audio = sd.rec(int(seconds * self.sample_rate),
                       samplerate=self.sample_rate, channels=1, dtype="float32")
        sd.wait()
        return audio[:, 0]

    def load_file(self, path):
        """Or load audio from a file, converted to mono 16000 Hz."""
        y, _ = librosa.load(path, sr=self.sample_rate, mono=True)
        return y

    def extract(self, y):
        """Cut the audio into 0.5 second windows and measure loudness of each."""
        results = []
        for i in range(0, len(y) - self.window_size + 1, self.window_size):
            chunk = y[i:i + self.window_size]
            loudness = float(np.sqrt(np.mean(chunk ** 2)))
            results.append({
                "time": round(i / self.sample_rate, 2),
                "amplitude": round(loudness, 5),
            })
        return results

    
    def analyze(self, features, silence_level=0.01, speech_level=0.012):
        """Label each window and summarise the whole session."""
        labeled = []
        for f in features:
            amp = f["amplitude"]
            if amp < silence_level:
                label = "silence"
            elif amp < speech_level:
                label = "noise"
            else:
                label = "speech"
            labeled.append({**f, "label": label})

        total = len(labeled)
        if total == 0:
            return {"windows": [], "summary": {}}

        counts = {"silence": 0, "noise": 0, "speech": 0}
        for w in labeled:
            counts[w["label"]] += 1

        amps = [w["amplitude"] for w in labeled]
        summary = {
            "silence_ratio": round(counts["silence"] / total, 2),
            "noise_ratio": round(counts["noise"] / total, 2),
            "speech_ratio": round(counts["speech"] / total, 2),
            "average_amplitude": round(sum(amps) / total, 5),
            "peak_amplitude": round(max(amps), 5),
        }
        return {"windows": labeled, "summary": summary}

    