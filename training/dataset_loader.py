from datasets import load_dataset


def load_go_emotions():
    dataset = load_dataset("go_emotions")
    return dataset 


from collections import Counter

label_counter = Counter()
dataset = load_dataset("go_emotions")
for item in dataset["train"]:
    for label in item["labels"]:
        label_counter[label] += 1

print(label_counter)