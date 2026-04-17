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
    # EMOTION STYLE (slightly improved)
    # -----------------------------------
    def get_emotion_style(self, emotion):

        styles = {
            "joy": {
                "tone": "warm and natural",
                "instruction": "Be positive but not overly excited."
            },
            "sadness": {
                "tone": "soft and caring",
                "instruction": "Be gentle, emotional, and supportive."
            },
            "anxiety": {
                "tone": "calm and grounding",
                "instruction": "Help the user slow down mentally."
            },
            "anger": {
                "tone": "calm and understanding",
                "instruction": "Validate feelings, do not argue."
            },
            "stress": {
                "tone": "supportive and simple",
                "instruction": "Keep responses short and calming."
            }
        }

        return styles.get(emotion, {
            "tone": "friendly and natural",
            "instruction": "Respond like a real human friend."
        })

    # -----------------------------------
    # EMOTIONAL VARIATION (FIXED + SMART)
    # -----------------------------------
    def emotional_variation(self, emotion):

        responses = {
            "sadness": [
                "I'm really sorry you're going through that… do you want to talk about it?",
                "That sounds really heavy… I'm here with you.",
                "I hear you… what part has been the hardest?"
            ],
            "neutral": [
                "Hey, I'm here with you.",
                "Hi 🙂 how are you feeling right now?"
            ],
            "anxiety": [
                "Take a breath with me… what’s going on?",
                "You're not alone in this."
            ]
        }

        return random.choice(responses.get(emotion, ["I'm here with you."]))

    # -----------------------------------
    # PROMPT BUILDER (IMPROVED FLOW)
    # -----------------------------------
    def build_prompt(self, user_input, emotion):

        history = ""

        for turn in self.chat_history[-self.max_history:]:
            history += f"User: {turn['user']}\nFriend: {turn['bot']}\n"

        style = self.get_emotion_style(emotion)

        prompt = f"""You are a close friend.

RULES:
- Do NOT behave like a robot or assistant
- NEVER give long explanations
- NEVER ask more than ONE question
- Sometimes DO NOT ask any question at all
- First respond emotionally, then optionally respond with a question
- Keep it natural and human

Tone: {style['tone']}
Instruction: {style['instruction']}

Conversation:
{history}

User: {user_input}

Friend:
"""

        return prompt

    # -----------------------------------
    # GENERATE RESPONSE (FIXED INTELLIGENCE FLOW)
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
                max_new_tokens=45,
                do_sample=True,
                temperature=0.55,   # 🔥 more stable = more intelligent
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
        # REMOVE LEAKS
        # -----------------------------------
        bad_phrases = [
            "User:", "Assistant:", "Friend:",
            "AI", "language model",
            "training data",
            "Task", "Step",
            "Instructions",
            "[user]", "[friend]"
        ]

        for phrase in bad_phrases:
            response = response.replace(phrase, "")

        response = response.strip()

        # -----------------------------------
        # INTELLIGENCE FIX (IMPORTANT)
        # -----------------------------------
        words = response.split()

        if len(words) < 4:
            response = self.emotional_variation(emotion)

        if emotion == "neutral" and len(words) > 14:
            response = self.emotional_variation("neutral")

        response = self.clean_response(response)

        if not response:
            response = "I'm here with you."

        self.chat_history.append({
            "user": user_input,
            "bot": response
        })

        return response