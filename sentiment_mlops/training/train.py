#Imports

import mlflow
import mlflow.pytorch
import torch
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

#Load and prepare the data

def load_and_prepare_data():
    print("Loading IMDB dataset...")
    dataset = load_dataset("imdb")

    # Use a small subset for faster training
    train_dataset = dataset["train"].shuffle(seed=42).select(range(2000))
    test_dataset = dataset["test"].shuffle(seed=42).select(range(500))

    return train_dataset, test_dataset

#Tokenization

def tokenize_data(train_dataset, test_dataset):
    print("Tokenizing data...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=256,
        )

    train_tokenized = train_dataset.map(tokenize_function, batched=True)
    test_tokenized = test_dataset.map(tokenize_function, batched=True)

    train_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    return train_tokenized, test_tokenized, tokenizer

#Metrix function

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}

#Main training function

def train():
    mlflow.set_experiment("sentiment-analysis")

    with mlflow.start_run():
        # Hyperparameters
        LEARNING_RATE = 2e-5
        EPOCHS = 2
        BATCH_SIZE = 16

        # Log hyperparameters to MLflow
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("model_name", "distilbert-base-uncased")
        mlflow.log_param("train_size", 2000)

        # Load data
        train_dataset, test_dataset = load_and_prepare_data()
        train_tokenized, test_tokenized, tokenizer = tokenize_data(
            train_dataset, test_dataset
        )

        # Load model
        print("Loading DistilBERT model...")
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=2,
        )

        # Training settings
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            logging_dir="./logs",
            logging_steps=50,
            report_to="none",
        )

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=test_tokenized,
            compute_metrics=compute_metrics,
        )

        # Train
        print("Starting training...")
        trainer.train()

        # Evaluate
        print("Evaluating model...")
        results = trainer.evaluate()
        print(f"Results: {results}")

        # Log metrics to MLflow
        mlflow.log_metric("accuracy", results["eval_accuracy"])
        mlflow.log_metric("f1", results["eval_f1"])

        # Save model to MLflow
        print("Saving model to MLflow...")
        mlflow.pytorch.log_model(model, "model")
        tokenizer.save_pretrained("./tokenizer")
        mlflow.log_artifacts("./tokenizer", artifact_path="tokenizer")

        print("Training complete!")
        print(f"Accuracy: {results['eval_accuracy']:.4f}")
        print(f"F1 Score: {results['eval_f1']:.4f}")


if __name__ == "__main__":
    train()

