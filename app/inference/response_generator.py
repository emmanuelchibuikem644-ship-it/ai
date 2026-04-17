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

    def generate(self, user_input):

        # --------------------------------
        # STEP 1: EMOTION DETECTION
        # --------------------------------
        emotions = self.emotion_model.predict_emotions(user_input)

        primary_emotion = emotions[0]["emotion"] if emotions else "neutral"
        primary_emotion = primary_emotion.lower().strip()

        # --------------------------------
        # STEP 2: GENERATE RESPONSE
        # --------------------------------
        response = self.dialog_model.generate_response(
            user_input,
            emotion=primary_emotion
        )

        # --------------------------------
        # STEP 3: SMART COPING STRATEGY
        # --------------------------------
        strategies = self.coping_strategies.get(primary_emotion, [])

        if primary_emotion in ["sadness", "stress", "anxiety"] and strategies:
            strategy = random.choice(strategies)
            response += f"\n\n💡 {strategy}"

        # --------------------------------
        # STEP 4: CLEAN FIRST
        # --------------------------------
        response = self.cleaner.clean(response)

        # --------------------------------
        # STEP 5: SAFETY FILTER LAST
        # --------------------------------
        response = self.safety_filter.filter_response(user_input, response)

        return {
            "emotion": primary_emotion,
            "response": response
        }