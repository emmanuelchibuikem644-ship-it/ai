from datasets import load_dataset

dataset = load_dataset("empathetic_dialogues")

def format_dialogue(example):
    return {
        "text": f"User: {example['utterance']}\nBot: {example['response']}"
    }

dataset = dataset.map(format_dialogue)

dataset = dataset.remove_columns(
    [col for col in dataset["train"].column_names if col != "text"]
)

dataset.save_to_disk("empathetic_dialogues_formatted")