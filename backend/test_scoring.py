import json
from core.scoring import EngagementScorer

with open("fused_data.json") as f:
    fused = json.load(f)

result = EngagementScorer().score_session(fused)

for s in result["segments"]:
    print(f"{s['start']:>4}-{s['end']:<4}s  score={s['score']:>5}  {s['level']}")

print()
print("Average:", result["average_score"], "->", result["overall_level"])

with open("engagement_scores.json", "w") as f:
    json.dump(result, f, indent=2)
print("Saved engagement_scores.json")