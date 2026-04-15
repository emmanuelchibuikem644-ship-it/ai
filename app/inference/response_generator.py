import json
import random

from inference.emotion_predictor import EmotionPredictor
from inference.llama_predictor import LlamaPredictor
from inference.safety_filter import SafetyFilter
from inference.conversation_memory import ConversationMemory
from inference.response_cleaner import ResponseCleaner

class ResponseGenerator:

    def __init__(self):

        self.emotion_model = EmotionPredictor()
        self.dialog_model = LlamaPredictor()
        self.safety_filter = SafetyFilter()
        self.memory = ConversationMemory()
        self.cleaner = ResponseCleaner()

        with open("data/coping_strategies.json", "r", encoding="utf-8") as f:
            self.coping_strategies = json.load(f)

    def get_primary_emotion(self, emotions):
        return emotions[0]["emotion"]

    def generate(self, user_input):

        # -----------------------------
        # STEP 1: EMOTION DETECTION
        # -----------------------------
        emotions = self.emotion_model.predict_emotions(user_input)

        primary_emotion = emotions[0]["emotion"] if emotions else "neutral"
        primary_emotion = primary_emotion.lower().strip()

        # -----------------------------
        # STEP 2: GENERATE RESPONSE
        # -----------------------------
        response = self.dialog_model.generate_response(
            user_input,
            emotion=primary_emotion
        )

        # -----------------------------
        # STEP 3: COPING STRATEGY
        # -----------------------------
        strategies = self.coping_strategies.get(primary_emotion, [])

        if strategies:   # ONLY pick if not empty
            strategy = random.choice(strategies)
            response = f"{response}\n\n Helpful tip: {strategy}"
        else:
            # SAFE fallback (prevents crash)
            response += "\n\nTry to stay calm and take things step by step."

        # -----------------------------
        # STEP 4: SAFETY FILTER (FINAL AUTHORITY)
        # -----------------------------
        response = self.safety_filter.filter_response(user_input, response)

        return {
            "emotion": primary_emotion,
            "response": response
        }