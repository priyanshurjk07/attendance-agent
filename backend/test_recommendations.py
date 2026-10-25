import json
from core.recommendations import RecommendationEngine

with open("engagement_scores.json") as f:
    result = json.load(f)

recs = RecommendationEngine().generate(result)

for r in recs:
    print(f"[{r['type'].upper():7}] {r['time']:>10}  {r['message']}")

with open("recommendations.json", "w") as f:
    json.dump(recs, f, indent=2)
print("Saved recommendations.json")