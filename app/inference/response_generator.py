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

    # --------------------------------
    # NEW: HUMAN-LIKE FLOW CONTROL
    # --------------------------------
    def humanize_response(self, response, emotion, user_input):

        # remove weird leftovers
        bad_patterns = ["USER:", "User:", "Assistant:", "Friend:", "AI:"]
        for p in bad_patterns:
            response = response.replace(p, "")

        response = response.strip()

        # --------------------------------
        # CONTROL QUESTIONS (VERY IMPORTANT)
        # --------------------------------
        question_count = response.count("?")

        if question_count > 1:
            response = response.split("?")[0] + "?"

        # --------------------------------
        # EMOTION FLOW CONTROL
        # --------------------------------
        if emotion == "sadness":
            if "?" not in response:
                response += " Do you want to talk about what happened?"

        elif emotion == "neutral":
            # avoid long robotic replies
            if len(response.split()) > 12:
                response = random.choice([
                    "Hey  what’s on your mind?",
                    "Hi, how are you feeling right now?",
                    "I’m here with you."
                ])

        # --------------------------------
        # REMOVE ROBOTIC LANGUAGE
        # --------------------------------
        robotic_phrases = [
            "how can I assist",
            "what can I do for you",
            "provide assistance",
            "support you with"
        ]

        for phrase in robotic_phrases:
            if phrase in response.lower():
                response = "I’m here with you."

        return response.strip()

    # --------------------------------
    # MAIN GENERATE FUNCTION
    # --------------------------------
    def generate(self, user_input):

        # --------------------------------
        # STEP 1: EMOTION DETECTION
        # --------------------------------
        emotions = self.emotion_model.predict_emotions(user_input)

        primary_emotion = emotions[0]["emotion"] if emotions else "neutral"
        primary_emotion = primary_emotion.lower().strip()

        # --------------------------------
        # STEP 2: GENERATE RESPONSE (MODEL)
        # --------------------------------
        response = self.dialog_model.generate_response(
            user_input,
            emotion=primary_emotion
        )

        # --------------------------------
        # STEP 3: HUMANIZE ( IMPORTANT)
        # --------------------------------
        response = self.humanize_response(response, primary_emotion, user_input)

        # --------------------------------
        # STEP 4: ADD COPING STRATEGY (SMART)
        # --------------------------------
        strategies = self.coping_strategies.get(primary_emotion, [])

        # only add sometimes (not always → more natural)
        if primary_emotion in ["sadness", "stress", "anxiety"] and strategies:
            if random.random() < 0.5:  # 50% chance
                strategy = random.choice(strategies)
                response += f"\n\n💡 {strategy}"

        # --------------------------------
        # STEP 5: CLEAN RESPONSE
        # --------------------------------
        response = self.cleaner.clean(response)

        # --------------------------------
        # STEP 6: SAFETY FILTER (FINAL)
        # --------------------------------
        response = self.safety_filter.filter_response(user_input, response)

        return {
            "emotion": primary_emotion,
            "response": response
        }