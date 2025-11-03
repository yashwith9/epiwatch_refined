"""
Train transformer models with CPU optimizations
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
    Trainer, TrainingArguments, EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

print("="*80)
print("EPIWATCH TRANSFORMER MODELS TRAINING (CPU OPTIMIZED)")
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

# Split
X_train, X_temp, y_train, y_temp = train_test_split(
    df['processed_text'], df['label'], test_size=0.3, random_state=42, stratify=df['label']
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.666, random_state=42, stratify=y_temp
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}\n")

# Custom Dataset
class EpidemicDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
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

# Models to train (CPU-friendly versions)
models_config = {
    "DistilBERT": "distilbert-base-multilingual-cased",  # Smallest
    "MuRIL": "google/muril-base-cased",
    "mBERT": "bert-base-multilingual-cased",
    "XLM-RoBERTa": "xlm-roberta-base"
}

results = {}

for model_name, model_path in models_config.items():
    print("="*80)
    print(f"TRAINING {model_name.upper()}")
    print("="*80)
    print()
    
    try:
        print(f"Loading {model_name} tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=2,
            ignore_mismatched_sizes=True
        )
        print(f"✓ Model loaded\n")
        
        # Create datasets
        print("Creating datasets...")
        train_dataset = EpidemicDataset(X_train, y_train, tokenizer)
        val_dataset = EpidemicDataset(X_val, y_val, tokenizer)
        test_dataset = EpidemicDataset(X_test, y_test, tokenizer)
        print(f"✓ Datasets created\n")
        
        # Training arguments (CPU optimized)
        training_args = TrainingArguments(
            output_dir=f'./results/{model_name.lower()}',
            num_train_epochs=2,  # Reduced for CPU
            per_device_train_batch_size=4,  # Small batch size for CPU
            per_device_eval_batch_size=8,
            learning_rate=2e-5,
            weight_decay=0.01,
            logging_dir=f'./logs/{model_name.lower()}',
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            save_total_limit=1,
            report_to="none",
            fp16=False,  # No mixed precision on CPU
            dataloader_num_workers=0,  # No multiprocessing on CPU
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # Train
        print(f"Training {model_name}...")
        start_time = time.time()
        trainer.train()
        train_time = time.time() - start_time
        print(f"✓ Training completed in {train_time:.2f}s\n")
        
        # Evaluate on test set
        print(f"Evaluating {model_name} on test set...")
        start_inference = time.time()
        predictions = trainer.predict(test_dataset)
        inference_time = (time.time() - start_inference) / len(X_test)
        
        metrics = compute_metrics(predictions)
        
        # Save results
        results[model_name] = {
            "accuracy": float(metrics['accuracy']),
            "precision": float(metrics['precision']),
            "recall": float(metrics['recall']),
            "f1": float(metrics['f1']),
            "train_time": float(train_time),
            "inference_time_per_sample": float(inference_time * 1000),  # in ms
            "model_type": "transformer",
            "model_path": model_path
        }
        
        print(f"\n{model_name} Results:")
        print(f"  - Accuracy: {metrics['accuracy']:.4f}")
        print(f"  - Precision: {metrics['precision']:.4f}")
        print(f"  - Recall: {metrics['recall']:.4f}")
        print(f"  - F1 Score: {metrics['f1']:.4f}")
        print(f"  - Training time: {train_time:.2f}s")
        print(f"  - Inference: {inference_time*1000:.2f}ms per sample\n")
        
        # Save model
        model_save_path = f"models/saved/{model_name.lower()}_best"
        trainer.save_model(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        print(f"✓ Model saved to {model_save_path}\n")
        
        # Clear memory
        del model, trainer, train_dataset, val_dataset, test_dataset
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        print(f"✗ Error training {model_name}: {str(e)}\n")
        results[model_name] = {
            "error": str(e),
            "status": "failed"
        }
        continue

# Load custom model results
print("="*80)
print("LOADING CUSTOM NEURAL NETWORK RESULTS")
print("="*80)
print()

custom_model_path = "models/saved/custom_best.pt"
if os.path.exists(custom_model_path):
    print("Loading trained custom model...")
    
    from models.custom_model import CustomEpiDetector, build_vocab, EpidemicDataset as CustomDataset
    from torch.utils.data import DataLoader
    
    # Build vocab
    vocab = build_vocab(X_train.tolist())
    
    # Create dataset
    test_dataset = CustomDataset(X_test.tolist(), y_test.tolist(), vocab)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Load model
    device = torch.device('cpu')
    model = CustomEpiDetector(len(vocab), embedding_dim=256, hidden_dim=128, num_layers=2).to(device)
    checkpoint = torch.load(custom_model_path, map_location=device)
    model.load_state_dict(checkpoint)
    
    # Evaluate
    model.eval()
    all_preds = []
    all_labels = []
    start_inference = time.time()
    
    with torch.no_grad():
        for batch_texts, batch_labels in test_loader:
            batch_texts, batch_labels = batch_texts.to(device), batch_labels.to(device)
            outputs, _ = model(batch_texts)
            preds = (outputs > 0.5).long().squeeze()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    inference_time = (time.time() - start_inference) / len(X_test)
    
    # Metrics
    results["Custom Neural Network"] = {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "precision": float(precision_score(all_labels, all_preds)),
        "recall": float(recall_score(all_labels, all_preds)),
        "f1": float(f1_score(all_labels, all_preds)),
        "train_time": 0.0,
        "inference_time_per_sample": float(inference_time * 1000),
        "model_type": "deep_learning"
    }
    
    print(f"✓ Custom Neural Network:")
    print(f"  - Accuracy: {results['Custom Neural Network']['accuracy']:.4f}")
    print(f"  - F1 Score: {results['Custom Neural Network']['f1']:.4f}")
    print()

# Save results
os.makedirs("results", exist_ok=True)
with open("results/transformer_metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print("="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)
print()
print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
print("-"*80)

successful_models = {k: v for k, v in results.items() if 'error' not in v}
for model_name, metrics in sorted(successful_models.items(), key=lambda x: x[1].get('f1', 0), reverse=True):
    print(f"{model_name:<25} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} "
          f"{metrics['recall']:<12.4f} {metrics['f1']:<12.4f}")

failed_models = {k: v for k, v in results.items() if 'error' in v}
if failed_models:
    print("\nFailed models:")
    for model_name in failed_models:
        print(f"  - {model_name}")

print()
print("✓ Results saved to results/transformer_metrics.json")
print("✓ All models trained! Ready to generate README with comparisons!")
