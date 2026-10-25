import json
from core.analytics import AnalyticsService

with open("engagement_scores.json") as f:
    result = json.load(f)

report = AnalyticsService().generate(result)

a = report["attendance"]
p = report["participation"]

print("ATTENDANCE")
print("  Marked present:", a["marked_present"], "| at:", a["marked_present_at"])
print()
print("PARTICIPATION")
print(f"  Visible: {p['visible_seconds']}s | Speaking: {p['speaking_seconds']}s | Focused: {p['focused_seconds']}s")
print(f"  Average score: {p['average_score']} ({p['overall_level']}) | Pattern: {p['consistency']}")
print()
print("TIMELINE")
for t in report["timeline"]:
    print(f"  {t['time']:>10}  {t['score']:>5}  {t['level']}")
print()
print("REFLECTION")
for n in report["reflection"]:
    print("  -", n)

with open("session_analytics.json", "w") as f:
    json.dump(report, f, indent=2)
print()
print("Saved session_analytics.json")