import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


class InsightGenerator:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        self.model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self.client = Groq(api_key=api_key) if api_key else None

    def fallback(self, result):
        """Rule-based suggestion, used when the AI is not available."""
        level = result["overall_level"]
        segs = result["segments"]
        speech = sum(s["speech_ratio"] for s in segs) / len(segs) if segs else 0
        face = sum(s["face_present_ratio"] for s in segs) / len(segs) if segs else 0

        if level == "high":
            insight = "Students are engaged and participating well."
            rec = "Maintain the current pace and keep the discussion going."
        elif face < 0.5:
            insight = "Many participants were not clearly visible or looking at the screen."
            rec = "Ask a quick question or run a short activity to regain attention."
        elif speech < 0.2:
            insight = "The session was mostly quiet with little interaction."
            rec = "Invite questions or run a quick poll to encourage participation."
        else:
            insight = "Engagement is moderate with some drops during the session."
            rec = "Add an interactive element to keep attention steady."
        return {"source": "rules", "insight": insight, "recommendation": rec}

    def generate(self, result):
        if self.client is None:
            print("No GROQ_API_KEY found, using rule-based suggestion.")
            return self.fallback(result)

        lines = []
        for s in result["segments"]:
            lines.append(f"{s['start']}-{s['end']}s: score {s['score']} ({s['level']}), "
                         f"face visible {int(s['face_present_ratio'] * 100)}%, "
                         f"speech {int(s['speech_ratio'] * 100)}%")
        prompt = (
            "Engagement data for a classroom session:\n"
            + "\n".join(lines)
            + f"\nAverage score: {result['average_score']} ({result['overall_level']}).\n"
            "Reply with exactly two lines:\n"
            "Insight: <one sentence about engagement>\n"
            "Recommendation: <one practical suggestion for the teacher>"
        )

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "You are an educational AI assistant. Be brief and professional."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
                max_tokens=800,
            )
            text = completion.choices[0].message.content or ""
            if not text.strip():
                raise ValueError("AI returned an empty answer")
            insight, rec = "", ""
            for line in text.splitlines():
                if line.lower().startswith("insight:"):
                    insight = line.split(":", 1)[1].strip()
                elif line.lower().startswith("recommendation:"):
                    rec = line.split(":", 1)[1].strip()
            if not insight:
                insight = text.strip()
            return {"source": "ai", "insight": insight, "recommendation": rec}
        except Exception as e:
            print(f"AI error: {e}")
            return self.fallback(result)