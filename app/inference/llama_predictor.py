import torch
import re
import random

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


class LlamaPredictor:

    def __init__(self, model_path="models/llama_mental_health", max_history=3):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        base_model = AutoModelForCausalLM.from_pretrained(
            "models/base_model",
            torch_dtype=torch.float32,
            device_map=None
        )

        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()

        self.chat_history = []
        self.max_history = max_history

    # -----------------------------------
    # CLEAN RESPONSE
    # -----------------------------------
    def clean_response(self, text):

        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            return "I'm here with you."

        # REMOVE weird continuation junk
        stop_tokens = ["User:", "Assistant:", "Friend:", "[", "]", "Conversation", "Example"]
        for token in stop_tokens:
            if token in text:
                text = text.split(token)[0]

        sentences = text.split(".")
        seen = []
        final = []

        for s in sentences:
            s = s.strip()
            if s and s not in seen:
                seen.append(s)
                final.append(s)

        text = ". ".join(final).strip()

        if not text.endswith((".", "!", "?")):
            text += "."

        return text

    # -----------------------------------
    # EMOTION STYLE
    # -----------------------------------
    def get_emotion_style(self, emotion):

        styles = {
            "joy": {
                "tone": "warm and light",
                "instruction": "Be positive but natural, not over-excited."
            },
            "sadness": {
                "tone": "soft and caring",
                "instruction": "Be gentle, understanding, and emotionally present."
            },
            "anxiety": {
                "tone": "calm and grounding",
                "instruction": "Slow things down and be reassuring."
            },
            "anger": {
                "tone": "calm and understanding",
                "instruction": "Acknowledge feelings without arguing."
            },
            "stress": {
                "tone": "supportive and simple",
                "instruction": "Keep it short and calming."
            }
        }

        return styles.get(emotion, {
            "tone": "friendly and natural",
            "instruction": "Respond like a real human friend."
        })

    # -----------------------------------
    # EMOTIONAL VARIATION (FALLBACK)
    # -----------------------------------
    def emotional_variation(self, emotion):

        responses = {
            "sadness": [
                "I'm really sorry you're feeling this way… do you want to talk about it?",
                "That sounds really heavy… I'm here with you.",
                "I hear you… what’s been on your mind?"
            ],
            "neutral": [
                "Hey  how are you feeling today?",
                "Hi, I'm here with you."
            ],
            "anxiety": [
                "That sounds overwhelming… what’s been on your mind?",
                "You're not alone in this."
            ]
        }

        return random.choice(responses.get(emotion, ["I'm here with you."]))

    # -----------------------------------
    # PROMPT BUILDER (VERY IMPORTANT)
    # -----------------------------------
    def build_prompt(self, user_input, emotion):

        history = ""

        for turn in self.chat_history[-self.max_history:]:
            history += f"User: {turn['user']}\nFriend: {turn['bot']}\n"

        style = self.get_emotion_style(emotion)

        prompt = f"""You are a close friend talking to someone you care about.

GOAL:
- Be emotionally supportive like a real friend
- Do NOT act like a therapist or assistant
- Do NOT give medical or professional advice

RULES:
- First acknowledge the feeling
- Then respond naturally like a human
- Keep replies SHORT (1–2 sentences)
- Ask at most ONE gentle question (optional)
- Sometimes do NOT ask a question
- Avoid repeating phrases
- Do NOT repeat the same sentence twice
- Do NOT respond with generic phrases like "what's on your mind?" repeatedly
- Always respond based on what the user actually said
- Give practical advice when appropriate
- Ask specific follow-up questions

STYLE:
Tone: {style['tone']}
Behavior: {style['instruction']}

Conversation:
{history}

User: {user_input}

Friend:
"""

        return prompt

    # -----------------------------------
    # GENERATE RESPONSE
    # -----------------------------------
    def generate_response(self, user_input, emotion="neutral"):

        prompt = self.build_prompt(user_input, emotion)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=True,
                temperature=0.55,
                top_p=0.85,
                repetition_penalty=1.4,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # -----------------------------------
        # SAFE EXTRACTION
        # -----------------------------------
        if "Friend:" in decoded:
            response = decoded.split("Friend:")[-1]
        else:
            response = decoded

        # -----------------------------------
        # REMOVE JUNK / LEAKS
        # -----------------------------------
        stop_tokens = ["User:", "Assistant:", "Friend:", "AI:", "Conversation"]
        for token in stop_tokens:
            if token in response:
                response = response.split(token)[0]

        bad_phrases = [
            "AI", "language model", "training data",
            "[user]", "[friend]", "[prosper]"
        ]

        for phrase in bad_phrases:
            response = response.replace(phrase, "")

        response = response.strip()

        # -----------------------------------
        # REMOVE ROBOTIC STARTS
        # -----------------------------------
        bad_starts = ["I hear you", "I understand", "I can relate"]

        for bad in bad_starts:
            if response.startswith(bad):
                response = self.emotional_variation(emotion)

        # -----------------------------------
        # CONTROL QUESTIONS
        # -----------------------------------
        if response.count("?") > 1:
            response = response.split("?")[0] + "?"

        # -----------------------------------
        # PREVENT REPEATING SAME RESPONSE
        # -----------------------------------
        if any(response == turn["bot"] for turn in self.chat_history):
            response = self.emotional_variation(emotion)

        # -----------------------------------
        # SMART FALLBACK
        # -----------------------------------
        words = response.split()

        if len(words) < 3 or response.lower() in ["i'm here with you.", "i am here with you."]:
            response = self.emotional_variation(emotion)

        # -----------------------------------
        # STOP OVER-TALKING
        # -----------------------------------
        if len(words) > 20:
            response = self.emotional_variation(emotion)

        response = self.clean_response(response)

        if not response:
            response = "I'm here with you."

        self.chat_history.append({
            "user": user_input,
            "bot": response
        })

        return response