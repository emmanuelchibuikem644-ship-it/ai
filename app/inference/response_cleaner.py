import re

class ResponseCleaner:

    def clean(self, text):

        # -----------------------------
        # FIX TOKEN ARTIFACTS
        # -----------------------------
        text = text.replace("_comma_", ",")
        text = text.replace("_period_", ".")

        # -----------------------------
        # REMOVE AI / SYSTEM JUNK
        # -----------------------------
        bad_patterns = [
            "User:", "Assistant:", "Friend:", "AI:",
            "Response:", "Reply:", "Task", "Step",
            "Instructions", "Example",
            "as an AI", "language model", "training data"
        ]

        for pattern in bad_patterns:
            if pattern in text:
                text = text.split(pattern)[0]

        # -----------------------------
        # REMOVE BRACKETS / DATASET NOISE
        # -----------------------------
        text = re.sub(r"\[.*?\]", "", text)

        # -----------------------------
        # REMOVE EXTRA SPACES
        # -----------------------------
        text = re.sub(r"\s+", " ", text).strip()

        # -----------------------------
        # REMOVE REPETITION (YOUR LOGIC)
        # -----------------------------
        sentences = text.split(".")
        seen = set()
        cleaned = []

        for s in sentences:
            s = s.strip()
            if s and s not in seen:
                cleaned.append(s)
                seen.add(s)

        text = ". ".join(cleaned).strip()

        # -----------------------------
        # LIMIT LENGTH (VERY IMPORTANT)
        # -----------------------------
        words = text.split()
        if len(words) > 25:
            text = " ".join(words[:25])

        # -----------------------------
        # FINAL SAFETY
        # -----------------------------
        if not text:
            return "I'm here with you."

        if not text.endswith((".", "!", "?")):
            text += "."

        return text