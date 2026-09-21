# ============ Step 1 — Imports ============

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)


# ============ Step 2 — Load dataset ============

print("Loading Dataset.......")

dataset = load_dataset(
    "csv",
    data_files="customer_reviews_1200.csv"
)

print(dataset)



# ============ Step 3 — Create train / validation / test splits ============

print("\nCreating dataset splits.......")

# Convert the integer label column into a Hugging Face ClassLabel.
# This allows us to use stratified splitting.

from datasets import ClassLabel

dataset["train"] = dataset["train"].cast_column(
    "label",
    ClassLabel(
        names=[
            "negative",
            "neutral",
            "positive"
        ]
    )
)

# First split:
# 80% training
# 20% temporary test/validation data

train_test = dataset["train"].train_test_split(
    test_size=0.20,
    seed=42,
    stratify_by_column="label"
)

train_dataset = train_test["train"]
test_dataset = train_test["test"]


# Split the remaining 20% into:
# 10% validation
# 10% final test

validation_test = test_dataset.train_test_split(
    test_size=0.50,
    seed=42,
    stratify_by_column="label"
)

validation_dataset = validation_test["train"]
test_dataset = validation_test["test"]


# Put everything into one DatasetDict

from datasets import DatasetDict

dataset = DatasetDict({
    "train": train_dataset,
    "validation": validation_dataset,
    "test": test_dataset
})


print("\nDataset sizes:")
print("Training:", len(dataset["train"]))
print("Validation:", len(dataset["validation"]))
print("Testing:", len(dataset["test"]))


# ============ Step 4 — Load DistilBERT tokenizer ============

print("\nLoading Tokenizer.......")

tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased"
)


# ============ Step 5 — Tokenize the reviews ============

def tokenize(batch):

    return tokenizer(
        batch["review"],
        truncation=True,
        padding="max_length",
        max_length=128
    )


print("\nTokenizing Dataset........")

tokenized_dataset = {}

for split in dataset:

    tokenized_dataset[split] = dataset[split].map(
        tokenize,
        batched=True
    )


# ============ Step 6 — Prepare labels and remove unnecessary columns ============

# Our CSV already has a "label" column.
# We need to rename it to "labels" because
# Hugging Face Trainer expects the target column to be called "labels".

for split in tokenized_dataset:

    tokenized_dataset[split] = tokenized_dataset[split].rename_column(
        "label",
        "labels"
    )


# Remove columns that the model does not need.

columns_to_remove = [
    "review",
    "sentiment",
    "category",
    "review_id"
]

for split in tokenized_dataset:

    tokenized_dataset[split] = tokenized_dataset[split].remove_columns(
        [
            column
            for column in columns_to_remove
            if column in tokenized_dataset[split].column_names
        ]
    )


# ============ Step 7 — Load DistilBERT ============

print("\nLoading DistilBERT...")

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",

    num_labels=3,

    id2label={
        0: "negative",
        1: "neutral",
        2: "positive"
    },

    label2id={
        "negative": 0,
        "neutral": 1,
        "positive": 2
    }
)


# ============ Step 8 — Evaluation metrics ============

def compute_metrics(eval_prediction):

    predictions, labels = eval_prediction

    # Convert model outputs into the predicted class.
    predictions = np.argmax(
        predictions,
        axis=1
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ============ Step 9 — Training configuration ============

training_args = TrainingArguments(

    output_dir="./sentiment_model",

    num_train_epochs=3,

    per_device_train_batch_size=8,

    per_device_eval_batch_size=8,

    learning_rate=2e-5,

    # Your Transformers version uses eval_strategy.
    eval_strategy="epoch",

    save_strategy="epoch",

    load_best_model_at_end=True,

    metric_for_best_model="f1",

    greater_is_better=True,

    logging_steps=100,

    report_to="none"
)


# ============ Step 10 — Create Trainer ============

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=tokenized_dataset["train"],

    eval_dataset=tokenized_dataset["validation"],

    compute_metrics=compute_metrics
)


# ============ Step 11 — Train ============

print("\nStarting training...")

trainer.train()


# ============ Step 12 — Evaluate on the test set ============

print("\nEvaluating on test set...")

test_results = trainer.evaluate(
    tokenized_dataset["test"]
)

print("\nTest results:")

for key, value in test_results.items():

    print(f"{key}: {value}")


# ============ Step 13 — Save the model ============

print("\nSaving model...")

trainer.save_model(
    "./sentiment_model"
)

tokenizer.save_pretrained(
    "./sentiment_model"
)

print("Model saved to ./sentiment_model")

