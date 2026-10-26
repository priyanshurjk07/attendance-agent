import os
import time
from dotenv import load_dotenv

load_dotenv()


class InsightGenerator:
    def __init__(self):
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.groq_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

    def fallback(self, result):
        """Rule-based suggestion, used when no AI is available."""
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
        return {"source": "rules", "provider": "rules",
                "insight": insight, "recommendation": rec}

    def _prompt(self, result):
        lines = []
        for s in result["segments"]:
            lines.append(f"{s['start']}-{s['end']}s: score {s['score']} ({s['level']}), "
                         f"face visible {int(s['face_present_ratio'] * 100)}%, "
                         f"speech {int(s['speech_ratio'] * 100)}%")
        return (
            "Engagement data for a classroom session:\n"
            + "\n".join(lines)
            + f"\nAverage score: {result['average_score']} ({result['overall_level']}).\n"
            "Reply with exactly two lines:\n"
            "Insight: <one sentence about engagement>\n"
            "Recommendation: <one practical suggestion for the teacher>"
        )

    def _parse(self, text, provider):
        insight, rec = "", ""
        for line in text.splitlines():
            if line.lower().startswith("insight:"):
                insight = line.split(":", 1)[1].strip()
            elif line.lower().startswith("recommendation:"):
                rec = line.split(":", 1)[1].strip()
        if not insight:
            insight = text.strip()
        return {"source": "ai", "provider": provider,
                "insight": insight, "recommendation": rec}

    def _ask_gemini(self, prompt):
        from google import genai
        client = genai.Client(api_key=self.gemini_key)
        resp = client.models.generate_content(model=self.gemini_model, contents=prompt)
        return resp.text or ""

    def _ask_groq(self, prompt):
        from groq import Groq
        client = Groq(api_key=self.groq_key)
        completion = client.chat.completions.create(
            model=self.groq_model,
            messages=[
                {"role": "system",
                 "content": "You are an educational AI assistant. Be brief and professional."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=800,
        )
        return completion.choices[0].message.content or ""

    def generate(self, result):
        prompt = self._prompt(result)

        providers = []
        if self.gemini_key:
            providers.append(("gemini", self._ask_gemini))
        if self.groq_key:
            providers.append(("groq", self._ask_groq))

        for name, ask in providers:
            for attempt in range(3):
                try:
                    text = ask(prompt)
                    if not text.strip():
                        raise ValueError("empty answer")
                    return self._parse(text, name)
                except Exception as e:
                    print(f"{name} error (try {attempt + 1}/3): {str(e)[:150]}")
                    if "503" in str(e) or "UNAVAILABLE" in str(e):
                        time.sleep(3)
                    else:
                        break
        print("No AI available, using rule-based suggestion.")
        return self.fallback(result)
