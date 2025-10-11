# model_training.py

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from typing import List


# -------------------- CUSTOM DATASET --------------------
class CommentsDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# -------------------- TRAINING CLASS --------------------
class ModelTrainer:
    def __init__(self, model_name="distilbert-base-uncased", num_labels=3):
        # Initialize tokenizer
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

        # Initialize model
        # used for first finetuning
        # self.model = DistilBertForSequenceClassification.from_pretrained(
#     model_name,
#     num_labels=num_labels
# )
        self.model = DistilBertForSequenceClassification.from_pretrained(
            r"C:\Users\Vineeth\Desktop\SVRE_P1\models\distillbert",  # previous checkpoint
            num_labels=num_labels
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(self.device)
    # -------------------- DATA PREPROCESSING --------------------
    def data_preprocessing(self, file_path: str, text_column: str = 'text', label_column: str = 'label', test_size: float = 0.2):
        df = pd.read_csv(file_path)

        # Encode labels
        le = LabelEncoder()
        df[label_column] = le.fit_transform(df[label_column])

        # Train/validation split
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df[text_column].tolist(),
            df[label_column].tolist(),
            test_size=test_size,
            random_state=42
        )

        # Create datasets
        train_dataset = CommentsDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = CommentsDataset(val_texts, val_labels, self.tokenizer)

        return train_dataset, val_dataset

    # -------------------- TRAINING --------------------
    def train(self, train_dataset, val_dataset, batch_size: int = 16, epochs: int = 3, lr: float = 5e-5):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        optimizer = AdamW(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_loss:.4f}")

            # Evaluate after each epoch
            self.evaluate(val_loader)

    # -------------------- EVALUATION --------------------
    def evaluate(self, val_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        print(f"Validation Accuracy: {acc:.4f}")
        return acc

    # -------------------- SAVE MODEL --------------------
    def save_model(self, save_path: str):
        # Used for first finetuning
        # self.model.save_pretrained(save_path)
        self.model.save_pretrained("C:/Users/Vineeth/Desktop/SVRE_P1/models/distillbert_finetuned_v2")
        # Used for first finetuning
        # self.tokenizer.save_pretrained(save_path)
        self.tokenizer.save_pretrained("C:/Users/Vineeth/Desktop/SVRE_P1/models/distillbert_finetuned_v2")

        print(f"Model saved to {save_path}")
