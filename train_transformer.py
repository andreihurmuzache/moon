from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from datasets import Dataset

# Exemplu date pentru antrenare
data = {
    "text": [
        "Gold prices rose significantly today.",
        "The market is bearish due to economic instability."
    ],
    "labels": [1, 0]
}

# Creare dataset
dataset = Dataset.from_dict(data)

# Tokenizare
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Model Transformer
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Antrenare model
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=10,
    save_total_limit=2,
    logging_dir="./logs",
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)
trainer.train()

# Salvare model
model.save_pretrained("./transformer_model")
tokenizer.save_pretrained("./transformer_model")
print("Model Transformer salvat în 'transformer_model/'.")
