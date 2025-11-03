"""
Pre-trained Transformer Models for Epidemic Detection
Fine-tuning: XLM-RoBERTa, mBERT, DistilBERT, MuRIL
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np
from tqdm import tqdm
import time


class EpidemicTransformerDataset(Dataset):
    """Dataset for transformer models"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
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
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class PretrainedEpiDetector:
    """
    Wrapper class for fine-tuning pre-trained transformer models
    """
    
    def __init__(self, model_name, num_labels=2, device=None):
        """
        Args:
            model_name: Name of pre-trained model
                - 'xlm-roberta-base'
                - 'bert-base-multilingual-cased'
                - 'distilbert-base-multilingual-cased'
                - 'google/muril-base-cased'
            num_labels: Number of output classes (2 for binary)
            device: torch device
        """
        self.model_name = model_name
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        ).to(self.device)
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        print(f"Model loaded on {self.device}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def prepare_dataloader(self, texts, labels, batch_size=16, max_length=512, shuffle=True):
        """Create DataLoader from texts and labels"""
        dataset = EpidemicTransformerDataset(texts, labels, self.tokenizer, max_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader
    
    def train_epoch(self, train_loader, optimizer, scheduler):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        predictions = []
        true_labels = []
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # Statistics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(true_labels, predictions)
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []
        probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                probs = torch.softmax(logits, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                probabilities.extend(probs[:, 1].cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='binary'
        )
        
        try:
            auc = roc_auc_score(true_labels, probabilities)
        except:
            auc = 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    
    def train(self, train_loader, val_loader, epochs=5, learning_rate=2e-5):
        """
        Train the model
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            epochs: Number of training epochs
            learning_rate: Learning rate
        """
        # Optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        best_val_f1 = 0.0
        
        print(f"\n{'='*60}")
        print(f"Training {self.model_name}")
        print(f"{'='*60}\n")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 60)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, scheduler)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validate
            val_metrics = self.validate(val_loader)
            self.val_losses.append(val_metrics['loss'])
            self.val_accuracies.append(val_metrics['accuracy'])
            
            # Print metrics
            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f}")
            print(f"F1-Score: {val_metrics['f1']:.4f} | AUC: {val_metrics['auc']:.4f}")
            
            # Save best model
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                model_save_name = self.model_name.replace('/', '_')
                save_path = f'models/saved/{model_save_name}_best.pt'
                torch.save(self.model.state_dict(), save_path)
                print(f"✓ Best model saved: {save_path}")
        
        print(f"\n{'='*60}")
        print(f"Training completed! Best F1-Score: {best_val_f1:.4f}")
        print(f"{'='*60}\n")
        
        return self.train_losses, self.val_losses
    
    def predict(self, texts, batch_size=16, max_length=512):
        """
        Generate predictions for new texts
        
        Args:
            texts: List of text strings
            batch_size: Batch size for inference
            max_length: Maximum sequence length
            
        Returns:
            predictions: Binary predictions (0 or 1)
            probabilities: Probability scores
        """
        self.model.eval()
        
        # Create dummy labels for dataset
        dummy_labels = [0] * len(texts)
        dataloader = self.prepare_dataloader(
            texts, dummy_labels, batch_size, max_length, shuffle=False
        )
        
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                probs = torch.softmax(logits, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                probabilities.extend(probs[:, 1].cpu().numpy())
        
        return np.array(predictions), np.array(probabilities)
    
    def measure_inference_time(self, sample_text, num_runs=100):
        """Measure average inference time"""
        self.model.eval()
        
        encoding = self.tokenizer(
            sample_text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(**encoding)
        
        # Measure
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.time()
                _ = self.model(**encoding)
                times.append(time.time() - start)
        
        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times)
        }


# Model configurations
MODEL_CONFIGS = {
    'xlm-roberta': {
        'name': 'xlm-roberta-base',
        'description': 'Cross-lingual RoBERTa - Excellent for multilingual tasks'
    },
    'mbert': {
        'name': 'bert-base-multilingual-cased',
        'description': 'Multilingual BERT - Supports 104 languages'
    },
    'distilbert': {
        'name': 'distilbert-base-multilingual-cased',
        'description': 'Distilled multilingual BERT - Faster, lighter'
    },
    'muril': {
        'name': 'google/muril-base-cased',
        'description': 'Multilingual Representations for Indian Languages'
    }
}


if __name__ == "__main__":
    print("Pre-trained Models for Epidemic Detection")
    print("=" * 60)
    
    for key, config in MODEL_CONFIGS.items():
        print(f"\n{key.upper()}")
        print(f"  Model: {config['name']}")
        print(f"  Description: {config['description']}")
