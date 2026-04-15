import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from dialogpt_metrics import DialogPTEvaluator


# -----------------------------
# Load trained model
# -----------------------------
model_path = "models/dialogpt_mental_health"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)


SYSTEM_PREFIX = (
    "You are a supportive mental wellness assistant. "
    "Respond with empathy, validation, and helpful coping strategies.\n"
)

# -----------------------------
# Load dataset
# -----------------------------
dataset = load_dataset("ShenLab/MentalChat16K")


def format_dialogue(example):

    user_input = example.get("input", "")
    response = example.get("output", "")
    instruction = example.get("instruction", "")

    if not user_input:
        user_input = instruction

    if not response:
        return None

    text = (
        SYSTEM_PREFIX +
        f"[EMOTION=neutral] User: {user_input}\nBot: {response}"
    )

    return {"text": text}


dataset = dataset.map(format_dialogue)

dataset = dataset.filter(lambda x: x is not None and "text" in x)  

dataset = dataset.map(
    format_dialogue,
    remove_columns=dataset["train"].column_names
)

dataset = dataset.flatten()

# -----------------------------
# Evaluation
# -----------------------------
evaluator = DialogPTEvaluator(model, tokenizer, device)

print("\nRunning evaluation...\n")

# Perplexity
perplexity = evaluator.compute_perplexity(dataset["validation"])
print("Perplexity:", perplexity)

# Generate responses
preds, refs = evaluator.generate_responses(dataset["validation"])

# Semantic similarity
similarity = evaluator.compute_semantic_similarity(preds, refs)
print("Semantic Similarity:", similarity)

# Quality
quality = evaluator.response_quality(preds)
print("Quality:", quality)

# Plot
evaluator.plot_metrics(perplexity, similarity, quality)