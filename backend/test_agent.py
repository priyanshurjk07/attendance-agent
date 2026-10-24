import json
from core.llm_insights import InsightGenerator

with open("engagement_scores.json") as f:
    result = json.load(f)

output = InsightGenerator().generate(result)

print("Source:        ", output["source"])
print("Insight:       ", output["insight"])
print("Recommendation:", output["recommendation"])

with open("ai_insight.json", "w") as f:
    json.dump(output, f, indent=2)
print("Saved ai_insight.json")