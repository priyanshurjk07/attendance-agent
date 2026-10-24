import json
from core.audio_analysis import AudioAnalyzer

analyzer = AudioAnalyzer()

print("Recording for 5 seconds... speak for 2-3 seconds, then stay quiet.")
y = analyzer.record(seconds=5)

features = analyzer.extract(y)
result = analyzer.analyze(features)

for w in result["windows"]:
    print(f"{w['time']:>4}s  {w['amplitude']:.4f}  {w['label']}")

print()
print("Summary:", result["summary"])

with open("audio_features.json", "w") as file:
    json.dump(result, file, indent=2)
print("Saved audio_features.json")