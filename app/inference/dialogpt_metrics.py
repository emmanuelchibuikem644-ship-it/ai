import torch
import numpy as np
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class DialogPTEvaluator:

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Semantic similarity model
        self.sim_model = SentenceTransformer("all-MiniLM-L6-v2")

    # -----------------------------------
    # Perplexity
    # -----------------------------------
    def compute_perplexity(self, dataset):

        self.model.eval()
        losses = []

        for example in dataset:
            inputs = self.tokenizer(
                example["text"],
                return_tensors="pt",
                truncation=True,
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss

            losses.append(loss.item())

        avg_loss = np.mean(losses)
        perplexity = np.exp(avg_loss)

        return perplexity

    # -----------------------------------
    # Semantic Similarity
    # -----------------------------------
    def compute_semantic_similarity(self, predictions, references):

        pred_embeddings = self.sim_model.encode(predictions)
        ref_embeddings = self.sim_model.encode(references)

        similarities = []

        for p, r in zip(pred_embeddings, ref_embeddings):
            sim = cosine_similarity([p], [r])[0][0]
            similarities.append(sim)

        return np.mean(similarities)

    # -----------------------------------
    # Response Quality Heuristics
    # -----------------------------------
    def response_quality(self, responses):

        lengths = [len(r.split()) for r in responses]
        avg_length = np.mean(lengths)

        diversity = len(set(responses)) / len(responses)

        return {
            "avg_length": avg_length,
            "diversity": diversity
        }

    # -----------------------------------
    # Generate model responses
    # -----------------------------------
    def generate_responses(self, dataset, num_samples=200):

        self.model.eval()

        predictions = []
        references = []

        for i, example in enumerate(dataset):
            if i >= num_samples:
                break

            input_text = example["text"]

            inputs = self.tokenizer.encode(
                input_text,
                return_tensors="pt"
            ).to(self.device)

            outputs = self.model.generate(
                inputs,
                max_new_tokens=50,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                top_k=50,
                top_p=0.95
            )

            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            predictions.append(generated)
            # Extract expected bot part if exists
            if "Bot:" in input_text:
                ref = input_text.split("Bot:")[-1].strip()
            else:
                ref = input_text

            references.append(ref)

        return predictions, references

    # -----------------------------------
    # Plot Metrics
    # -----------------------------------
    def plot_metrics(self, perplexity, similarity, quality):

        names = ["Perplexity", "Semantic Similarity", "Avg Length", "Diversity"]
        values = [
            perplexity,
            similarity,
            quality["avg_length"],
            quality["diversity"]
        ]

        plt.figure()
        plt.bar(names, values)
        plt.title("DialogGPT Evaluation Metrics")
        plt.ylabel("Scores")

        plt.savefig("dialogpt_metrics.png")
        plt.show()