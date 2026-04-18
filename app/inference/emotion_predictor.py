import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class EmotionPredictor:

    def __init__(self, model_path="models/emotion_classifier", threshold=0.30):

        self.model_path = model_path
        self.threshold = threshold

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print("Emotion Predictor running on:", self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path
        )

        self.model.to(self.device)
        self.model.eval()

        self.id2label = self.model.config.id2label

    # -----------------------------------
    # NEW: TEXT NORMALIZATION (IMPORTANT)
    # -----------------------------------
    def normalize_text(self, text):
        return text.lower().strip()

    def predict_emotions(self, text):

        text = self.normalize_text(text)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits

        # 🔥 CHANGE: softmax instead of sigmoid (more stable)
        probs = torch.softmax(logits, dim=-1)

        probs = probs.squeeze().cpu().numpy()

        predicted_emotions = []

        for i, prob in enumerate(probs):

            if prob >= self.threshold:
                emotion = self.id2label[i]

                predicted_emotions.append(
                    {
                        "emotion": emotion,
                        "confidence": float(prob)
                    }
                )

        # -----------------------------------
        # FIX: ALWAYS RETURN BEST EMOTION
        # -----------------------------------
        if len(predicted_emotions) == 0:

            max_index = probs.argmax()

            predicted_emotions.append(
                {
                    "emotion": self.id2label[max_index],
                    "confidence": float(probs[max_index])
                }
            )

        # -----------------------------------
        # NEW: SORT BY CONFIDENCE
        # -----------------------------------
        predicted_emotions = sorted(
            predicted_emotions,
            key=lambda x: x["confidence"],
            reverse=True
        )

        return predicted_emotions


if __name__ == "__main__":

    predictor = EmotionPredictor()

    while True:

        text = input("\nEnter a message (or 'quit'): ")

        if text.lower() == "quit":
            break

        emotions = predictor.predict_emotions(text)

        print("\nDetected emotions:")

        for e in emotions:
            print(f"{e['emotion']} ({e['confidence']:.2f})")