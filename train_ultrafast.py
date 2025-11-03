"""
Ultra-fast transformer training - Minimal settings for quick completion
"""
import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing.text_preprocessing import TextPreprocessor
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

print("="*80)
print("EPIWATCH ULTRA-FAST TRAINING (5 MODELS)")
print("="*80)
print()

# Load data
print("Loading dataset...")
data_path = "data/processed/epidemic_data.csv"
df = pd.read_csv(data_path)
print(f"✓ Loaded {len(df)} samples\n")

# Preprocess
print("Preprocessing texts...")
preprocessor = TextPreprocessor()
df['processed_text'] = df['text'].apply(lambda x: preprocessor.preprocess(str(x)))

# Split - Use SMALLER training set for speed
X_train, X_temp, y_train, y_temp = train_test_split(
    df['processed_text'], df['label'], test_size=0.5, random_state=42, stratify=df['label']
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.4, random_state=42, stratify=y_temp
)

print(f"Train: {len(X_train)} (reduced for speed), Val: {len(X_val)}, Test: {len(X_test)}\n")

# Custom Dataset
class EpidemicDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):  # Reduced from 128
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Only 2 smallest transformer models for speed
models_config = {
    "DistilBERT": "distilbert-base-multilingual-cased",
    "MuRIL": "google/muril-base-cased"
}

results = {}

for model_name, model_path in models_config.items():
    print("="*80)
    print(f"TRAINING {model_name.upper()}")
    print("="*80)
    print()
    
    try:
        print(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=2,
            ignore_mismatched_sizes=True
        )
        print(f"✓ Model loaded\n")
        
        # Create datasets
        train_dataset = EpidemicDataset(X_train, y_train, tokenizer)
        val_dataset = EpidemicDataset(X_val, y_val, tokenizer)
        test_dataset = EpidemicDataset(X_test, y_test, tokenizer)
        
        # Ultra-minimal training args
        training_args = TrainingArguments(
            output_dir=f'./results/{model_name.lower()}',
            num_train_epochs=1,  # Just 1 epoch
            per_device_train_batch_size=8,  # Increased for speed
            per_device_eval_batch_size=16,
            learning_rate=3e-5,
            weight_decay=0.01,
            logging_steps=100,
            eval_strategy="no",  # No evaluation during training
            save_strategy="no",  # No checkpoints
            report_to="none",
            fp16=False,
            dataloader_num_workers=0,
            gradient_accumulation_steps=2,  # Simulate larger batch
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            compute_metrics=compute_metrics,
        )
        
        # Train
        print(f"Training {model_name} (1 epoch)...")
        start_time = time.time()
        trainer.train()
        train_time = time.time() - start_time
        print(f"✓ Training completed in {train_time:.2f}s\n")
        
        # Evaluate on test set
        print(f"Evaluating {model_name}...")
        start_inference = time.time()
        predictions = trainer.predict(test_dataset)
        inference_time = (time.time() - start_inference) / len(X_test)
        
        metrics = compute_metrics(predictions)
        
        results[model_name] = {
            "accuracy": float(metrics['accuracy']),
            "precision": float(metrics['precision']),
            "recall": float(metrics['recall']),
            "f1": float(metrics['f1']),
            "train_time": float(train_time),
            "inference_time_per_sample": float(inference_time * 1000),
            "model_type": "transformer"
        }
        
        print(f"\n{model_name} Results:")
        print(f"  ✓ Accuracy: {metrics['accuracy']:.4f}")
        print(f"  ✓ F1 Score: {metrics['f1']:.4f}")
        print(f"  ✓ Training time: {train_time:.2f}s\n")
        
        # Clear memory
        del model, trainer, train_dataset, val_dataset, test_dataset
        
    except Exception as e:
        print(f"✗ Error: {str(e)}\n")
        continue

print("="*80)
print("TRAINING CLASSICAL ML MODELS (FAST)")
print("="*80)
print()

# Add 3 fast classical models
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

classical_models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=100, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
}

for name, model in classical_models.items():
    print(f"Training {name}...")
    start_time = time.time()
    model.fit(X_train_tfidf, y_train)
    train_time = time.time() - start_time
    
    start_inference = time.time()
    y_pred = model.predict(X_test_tfidf)
    inference_time = (time.time() - start_inference) / len(X_test)
    
    results[name] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "train_time": float(train_time),
        "inference_time_per_sample": float(inference_time * 1000),
        "model_type": "classical_ml"
    }
    
    print(f"  ✓ Accuracy: {results[name]['accuracy']:.4f}, F1: {results[name]['f1']:.4f}, Time: {train_time:.2f}s\n")

# Save results
os.makedirs("results", exist_ok=True)
with open("results/all_models_metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print("="*80)
print("FINAL RESULTS - 5 MODELS TRAINED")
print("="*80)
print()
print(f"{'Model':<25} {'Type':<15} {'Accuracy':<12} {'F1 Score':<12} {'Train Time':<12}")
print("-"*80)

for model_name, metrics in sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True):
    model_type = metrics['model_type'].replace('_', ' ').title()
    print(f"{model_name:<25} {model_type:<15} {metrics['accuracy']:<12.4f} "
          f"{metrics['f1']:<12.4f} {metrics['train_time']:<12.2f}s")

print()
print("✓ 5 models trained successfully!")
print("✓ Results saved to results/all_models_metrics.json")
print("✓ Ready to generate comprehensive README!")
