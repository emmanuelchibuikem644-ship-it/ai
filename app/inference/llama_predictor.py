import torch
import re

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
    # EMOTION STYLE
    # -----------------------------------
    def get_emotion_style(self, emotion):

        styles = {
            "joy": {
                "tone": "warm and natural",
                "instruction": "Respond like a happy friend, calm and real."
            },
            "sadness": {
                "tone": "soft and caring",
                "instruction": "Be gentle and emotionally supportive like a close friend."
            },
            "anxiety": {
                "tone": "calm and grounding",
                "instruction": "Help the user relax and reduce overthinking."
            },
            "anger": {
                "tone": "calm and understanding",
                "instruction": "Do not argue, just validate feelings."
            },
            "stress": {
                "tone": "supportive and simple",
                "instruction": "Give short calming responses only."
            }
        }

        return styles.get(emotion, {
            "tone": "friendly and natural",
            "instruction": "Respond like a normal human friend."
        })

    # -----------------------------------
    # PROMPT BUILDER (FIXED HUMAN MODE)
    # -----------------------------------
    def build_prompt(self, user_input, emotion):

        history = ""

        for turn in self.chat_history[-self.max_history:]:
            history += f"User: {turn['user']}\nFriend: {turn['bot']}\n"

        style = self.get_emotion_style(emotion)

        prompt = f"""You are a close friend talking to someone you care about.

RULES:
- You are NOT a technical assistant
- NEVER talk about tech, tasks, or helping with problems
- NEVER say "I can help with..." or "tech issues"
- Be warm, calm, and human
- Keep replies SHORT (1–2 sentences)
- Ask at most ONE gentle question (or none)
- Focus on feelings, not fixing

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
                max_new_tokens=45,
                do_sample=True,
                temperature=0.6,
                top_p=0.85,
                repetition_penalty=1.35,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # -----------------------------------
        # SAFE EXTRACTION
        # -----------------------------------
        if "Friend:" in decoded:
            response = decoded.split("Friend:")[-1]
        elif "Assistant:" in decoded:
            response = decoded.split("Assistant:")[-1]
        else:
            response = decoded

        # -----------------------------------
        #  NEW: CUT WEIRD CONTINUATIONS
        # -----------------------------------
        stop_tokens = ["[", "]", "Conversation", "Example", "The conversation"]
        for token in stop_tokens:
            if token in response:
                response = response.split(token)[0]

        # -----------------------------------
        # REMOVE BAD PATTERNS (INCLUDING TECH + DATASET LEAKS)
        # -----------------------------------
        bad_phrases = [
            "User:", "Assistant:", "Friend:",
            "AI", "as an AI", "language model",
            "training data", "Task", "Step",
            "Instructions", "Response example",
            "Reply:",

            # REMOVE TECH BEHAVIOR
            "tech", "technical", "issue",
            "assist you", "help you with",
            "I can help", "support with",

            # REMOVE DATASET GARBAGE
            "[user]", "[friend]", "[prosper]",
            "office", "meeting", "visit",
            "pandemic", "conversation continues"
        ]

        for phrase in bad_phrases:
            response = response.replace(phrase, "")

        # -----------------------------------
        # CUT LONG JUNK
        # -----------------------------------
        for word in ["Task", "Step", "1.", "2.", "3."]:
            if word in response:
                response = response.split(word)[0]

        response = response.strip()

        # -----------------------------------
        # LIMIT QUESTIONS
        # -----------------------------------
        if response.count("?") > 1:
            response = response.split("?")[0] + "?"

        # -----------------------------------
        #  FINAL EMOTION CONTROL
        # -----------------------------------
        if emotion == "sadness":
            response = "I'm really sorry you're feeling this way… do you want to talk about what happened?"

        if emotion == "neutral" and len(response.split()) > 12:
            response = "Hey  how are you doing?"

        response = self.clean_response(response)

        if not response:
            response = "I'm here with you."

        self.chat_history.append({
            "user": user_input,
            "bot": response
        })

        return response