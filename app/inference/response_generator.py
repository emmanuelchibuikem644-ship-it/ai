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
    # HUMAN-LIKE FLOW CONTROL (IMPROVED)
    # --------------------------------
    def humanize_response(self, response, emotion, user_input):

        bad_patterns = ["USER:", "User:", "Assistant:", "Friend:", "AI:"]
        for p in bad_patterns:
            response = response.replace(p, "")

        response = response.strip()

        # limit question spam
        if response.count("?") > 1:
            response = response.split("?")[0] + "?"

        # -------------------------------
        # SADNESS HANDLING
        # -------------------------------
        if emotion == "sadness":

            if len(response.split()) < 6:
                response = random.choice([
                    "That sounds really tough… I’m here with you.",
                    "I’m really sorry you're feeling this way.",
                    "That must have been hard…"
                ])

            if "?" not in response:
                response += " Do you want to talk more about it?"

        # -------------------------------
        # ANXIETY HANDLING
        # -------------------------------
        elif emotion == "anxiety":

            if len(response.split()) < 6:
                response = random.choice([
                    "That sounds overwhelming… take it one step at a time.",
                    "I hear you… things can feel really heavy sometimes.",
                    "You’re not alone in this."
                ])

        # -------------------------------
        # NEUTRAL HANDLING
        # -------------------------------
        elif emotion == "neutral":

            if len(response.split()) > 14 or len(response.split()) < 4:
                response = random.choice([
                    "Hey… how are you feeling right now?",
                    "I’m here with you.",
                    "What’s on your mind?"
                ])

        # -------------------------------
        # REMOVE ROBOTIC LANGUAGE
        # -------------------------------
        robotic_phrases = [
            "how can I assist",
            "what can I do for you",
            "provide assistance",
            "support you with",
            "I can help you with"
        ]

        for phrase in robotic_phrases:
            if phrase.lower() in response.lower():
                response = "I’m here with you."

        # -------------------------------
        # LIMIT LONG RESPONSES
        # -------------------------------
        if len(response.split()) > 22:
            response = "I hear you… tell me more about that."

        return response.strip()

    # --------------------------------
    # MAIN GENERATE FUNCTION
    # --------------------------------
    def generate(self, user_input):

        # -------------------------------
        # STEP 1: EMOTION DETECTION
        # -------------------------------
        emotions = self.emotion_model.predict_emotions(user_input)

        primary_emotion = emotions[0]["emotion"] if emotions else "neutral"
        primary_emotion = primary_emotion.lower().strip()

        # -------------------------------
        # STEP 2: GET MEMORY CONTEXT  (NEW IMPROVEMENT)
        # -------------------------------
        history = self.memory.get_history()
        context = "\n".join([f"{h['user']} -> {h['bot']}" for h in history])

        # -------------------------------
        # STEP 3: GENERATE RESPONSE
        # -------------------------------
        try:
            response = self.dialog_model.generate_response(
                user_input,
                emotion=primary_emotion,
                context=context  #  NOW USING MEMORY
            )
        except TypeError:
            # fallback if model doesn't accept context yet
            response = self.dialog_model.generate_response(
                user_input,
                emotion=primary_emotion
            )

        # -------------------------------
        # STEP 4: FALLBACK IF MODEL IS WEAK
        # -------------------------------
        if not response or len(response.split()) < 3:
            response = random.choice([
                "I hear you… tell me more.",
                "I’m here with you.",
                "Do you want to talk about it?"
            ])

        # -------------------------------
        # STEP 5: HUMANIZE RESPONSE
        # -------------------------------
        response = self.humanize_response(response, primary_emotion, user_input)

        # -------------------------------
        # STEP 6: MEMORY UPDATE 🔥
        # -------------------------------
        self.memory.add_turn(user_input, response)

        # -------------------------------
        # STEP 7: COPING STRATEGY (FIXED SPAM ISSUE)
        # -------------------------------
        strategies = self.coping_strategies.get(primary_emotion, [])

        if primary_emotion in ["sadness", "stress", "anxiety"] and strategies:
            if random.random() < 0.3:  # reduced from 40% → 30%
                strategy = random.choice(strategies)
                response += f"\n\n💡 {strategy}"

        # -------------------------------
        # STEP 8: CLEAN RESPONSE
        # -------------------------------
        response = self.cleaner.clean(response)

        # -------------------------------
        # STEP 9: SAFETY FILTER
        # -------------------------------
        response = self.safety_filter.filter_response(user_input, response)

        return {
            "emotion": primary_emotion,
            "response": response
        }