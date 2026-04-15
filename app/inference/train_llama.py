# train_llama.py
# Install unsloth library


import torch

from datasets import load_dataset, concatenate_datasets
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer


# -----------------------------
# LOAD MODEL (UNSLOTH OPTIMIZED)
# -----------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3.2-1b",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True  # VERY IMPORTANT (low VRAM)
)

# Enable LoRA fine-tuning
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "v_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True
)


# -----------------------------
# LOAD DATASETS
# -----------------------------
mentalchat = load_dataset("ShenLab/MentalChat16K", split="train")


# -----------------------------
# FORMAT DATA
# -----------------------------
def format_mentalchat(example):
    user = example.get("input", "")
    response = example.get("output", "")

    if not user or not response:
        return None

    return {
        "text": f"### Instruction:\n{user}\n\n### Response:\n{response}"
    }


# Apply formatting (FIXED SAFE VERSION)
mentalchat = mentalchat.map(format_mentalchat)

# remove None safely
mentalchat = mentalchat.filter(lambda x: x is not None)


# -----------------------------
# TRAINING CONFIG
# -----------------------------
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=mentalchat,
    dataset_text_field="text",
    max_seq_length=2048,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        num_train_epochs=3,
        max_steps=1000,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        output_dir="models/llama_mental_health",
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42
    ),
)


# -----------------------------
# TRAIN
# -----------------------------
trainer.train()


# -----------------------------
# SAVE MODEL
# -----------------------------
model.save_pretrained("models/llama_mental_health")
tokenizer.save_pretrained("models/llama_mental_health")