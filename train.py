# train.py
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model
from config import *
import torch

def train_model():
    print("🚀 Loading dataset...")

    # Load full IMDB dataset and shuffle with fixed seed
    full_dataset = load_dataset(DATASET_NAME, split="train").shuffle(seed=42)

    # Filter and take 2500 negative (label=0) and 2500 positive (label=1) reviews
    neg_examples = full_dataset.filter(lambda x: x["label"] == 0).select(range(2500))
    pos_examples = full_dataset.filter(lambda x: x["label"] == 1).select(range(2500))

    # Combine and shuffle again
    dataset = concatenate_datasets([neg_examples, pos_examples]).shuffle(seed=42)

    # Optional: filter out invalid texts
    dataset = dataset.filter(lambda x: x["text"] is not None and len(x["text"]) > 10)

    # 🔍 Debug: Check label balance
    sample_labels = [dataset[i]["label"] for i in range(min(500, len(dataset)))]
    pos_count = sum(1 for x in sample_labels if x == 1)
    neg_count = sum(1 for x in sample_labels if x == 0)
    print(f"📊 Dataset Balance (first 500): Positive={pos_count}, Negative={neg_count}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True
        )

    print("🧹 Preprocessing dataset...")
    dataset = dataset.map(preprocess, batched=True)

    # Model
    print("🧠 Loading base model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )

    # LoRA Configuration
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_lin", "v_lin"],  # Correct for DistilBERT
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        logging_dir=f"{CHECKPOINT_DIR}/logs",
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        report_to="none",  # Disable external loggers
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,  # Important for PEFT + custom datasets
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    print("🎓 Starting training...")
    trainer.train()

    print("💾 Saving model with correct label mapping...")
    model.config.id2label = {0: "Negative", 1: "Positive"}
    model.config.label2id = {"Negative": 0, "Positive": 1}
    model.save_pretrained(CHECKPOINT_DIR)
    tokenizer.save_pretrained(CHECKPOINT_DIR)
    print(f"🎉 Model saved to {CHECKPOINT_DIR}")

if __name__ == "__main__":
    train_model()