import torch
import json
import numpy as np
import torch.nn.functional as F

from datasets import Dataset
from sklearn.utils.class_weight import compute_class_weight

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    DataCollatorWithPadding,
    TrainingArguments
)

from dataset_loader import load_go_emotions
from preprocessing import preprocess_dataset, oversample_dataset, load_label_schema
from metrics import compute_metrics


# -----------------------------
# Load label mappings
# -----------------------------
def load_label_mappings():

    with open("data/labels.json", "r") as f:

        labels_data = json.load(f)

    id2label = {int(k): v for k, v in labels_data["id2label"].items()}

    label2id = labels_data["label2id"]

    return id2label, label2id


# -----------------------------
# Compute class weights
# -----------------------------
def compute_class_weights(dataset, num_labels):

    labels = np.array(dataset["labels"])

    weights = []

    for i in range(num_labels):

        column = labels[:, i]

        weight = compute_class_weight(
            class_weight="balanced",
            classes=np.array([0, 1]),
            y=column
        )[1]

        weights.append(weight)

    return torch.tensor(weights)


# -----------------------------
# Data collator for multi-label
# -----------------------------
class MultiLabelDataCollator(DataCollatorWithPadding):

    def __call__(self, features):

        labels = [torch.tensor(f["labels"], dtype=torch.float) for f in features]

        batch = super().__call__(features)

        batch["labels"] = torch.stack(labels)

        return batch


# -----------------------------
# Custom Trainer with weighted loss
# -----------------------------
class WeightedTrainer(Trainer):

    def __init__(self, class_weights=None, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.class_weights = class_weights


    def compute_loss(self, model, inputs, return_outputs=False):

        labels = inputs.pop("labels")

        outputs = model(**inputs)

        logits = outputs.logits

        loss = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=self.class_weights.to(logits.device)
        )

        return (loss, outputs) if return_outputs else loss


# -----------------------------
# Main training pipeline
# -----------------------------
def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)

    dataset = load_go_emotions()

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    encoded_dataset = preprocess_dataset(dataset, tokenizer)

    encoded_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    label_schema = load_label_schema()

    id2label, label2id = load_label_mappings()

    # -----------------------------
    # Dataset diagnostics
    # -----------------------------
    labels = np.array(encoded_dataset["train"]["labels"])

    print("\nEmotion Distribution:\n")

    for i, emotion in enumerate(label_schema):

        print(emotion, int(labels[:, i].sum()))

    # -----------------------------
    # Oversampling
    # -----------------------------
    train_list = oversample_dataset(
        encoded_dataset["train"].to_list(),
        label_schema
    )

    train_dataset = Dataset.from_list(train_list)

    # -----------------------------
    # Compute class weights
    # -----------------------------
    class_weights = compute_class_weights(
        encoded_dataset["train"],
        num_labels=len(label_schema)
    )

    # -----------------------------
    # Load model
    # -----------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(label_schema),
        id2label=id2label,
        label2id=label2id,
        problem_type="multi_label_classification"
    )

    model.to(device)

    # -----------------------------
    # Training arguments
    # -----------------------------
    training_args = TrainingArguments(

        output_dir="./models",

        evaluation_strategy="epoch",
        save_strategy="epoch",

        learning_rate=2e-5,

        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,

        num_train_epochs=5,

        weight_decay=0.01,

        logging_dir="./logs",
        logging_steps=50,

        fp16=torch.cuda.is_available(),

        load_best_model_at_end=True,

        metric_for_best_model="macro_f1",

        greater_is_better=True 
    )

    trainer = WeightedTrainer(

        model=model,
        args=training_args,

        train_dataset=train_dataset,
        eval_dataset=encoded_dataset["validation"],

        tokenizer=tokenizer,

        data_collator=MultiLabelDataCollator(tokenizer),

        compute_metrics=compute_metrics,

        class_weights=class_weights
    )

    trainer.train()

    results = trainer.evaluate()

    print("\nFinal Evaluation Results:\n", results)

    trainer.save_model("models/emotion_classifier_v2")

    tokenizer.save_pretrained("models/emotion_classifier_v2")


if __name__ == "__main__":

    print("CUDA Available:", torch.cuda.is_available())

    if torch.cuda.is_available():

        print("GPU:", torch.cuda.get_device_name(0))

    main()