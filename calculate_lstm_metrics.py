"""
Calculate detailed metrics for Custom LSTM+Attention model
Including: Accuracy, Precision, Recall, F1-Score, ROC-AUC
"""

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, 
    classification_report
)
from sklearn.model_selection import train_test_split
import sys
import os

# Add project root to path
sys.path.insert(0, 'c:\\Users\\Bruger\\OneDrive\\Desktop\\NLP')

from src.models.custom_model import CustomEpiDetector
from src.preprocessing.text_preprocessing import TextPreprocessor
import json

print("=" * 80)
print("📊 CUSTOM LSTM+ATTENTION MODEL - DETAILED METRICS")
print("=" * 80)

# Load data
print("\n📁 Loading dataset...")
data_path = 'data/processed/epidemic_data.csv'
df = pd.read_csv(data_path)
print(f"✓ Loaded {len(df)} samples")

# Split data
X = df['text'].values
y = df['label'].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}")

# Load vocabulary
print("\n📚 Loading vocabulary...")
with open('models/saved/vocab.json', 'r', encoding='utf-8') as f:
    vocab = json.load(f)
print(f"✓ Vocabulary size: {len(vocab)}")

# Preprocess text
print("\n🔄 Preprocessing text...")
preprocessor = TextPreprocessor()

def text_to_indices(text, vocab, max_len=100):
    """Convert text to indices"""
    tokens = preprocessor.preprocess(text).split()
    indices = [vocab.get(token, vocab.get('<UNK>', 0)) for token in tokens]
    
    # Pad or truncate
    if len(indices) < max_len:
        indices += [vocab.get('<PAD>', 0)] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    
    return indices

X_test_indices = [text_to_indices(text, vocab) for text in X_test]
X_test_tensor = torch.LongTensor(X_test_indices)
y_test_tensor = torch.FloatTensor(y_test)

print(f"✓ Test data shape: {X_test_tensor.shape}")

# Load model
print("\n🤖 Loading trained model...")
device = torch.device('cpu')
model = CustomEpiDetector(
    vocab_size=len(vocab),
    embedding_dim=256,
    hidden_dim=128,
    num_layers=2,
    dropout=0.3
)

checkpoint = torch.load('models/saved/custom_best.pt', map_location=device)
model.load_state_dict(checkpoint)  # Load directly, not from dict
model.to(device)
model.eval()
print("✓ Model loaded successfully")

# Make predictions
print("\n🔮 Generating predictions...")
with torch.no_grad():
    outputs, attention_weights = model(X_test_tensor)
    probabilities = outputs.squeeze().numpy()
    predictions = (probabilities > 0.5).astype(int)

print(f"✓ Generated {len(predictions)} predictions")

# Calculate metrics
print("\n📊 Calculating metrics...")

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)
f1 = f1_score(y_test, predictions, zero_division=0)

# ROC-AUC requires probability scores
try:
    roc_auc = roc_auc_score(y_test, probabilities)
except:
    roc_auc = 0.0

# Confusion matrix
cm = confusion_matrix(y_test, predictions)
tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

# Specificity
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

print("\n" + "=" * 80)
print("✅ CUSTOM LSTM+ATTENTION MODEL - COMPLETE METRICS")
print("=" * 80)

print("\n📈 CLASSIFICATION METRICS")
print("-" * 80)
print(f"Accuracy:       {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision:      {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall:         {recall:.4f} ({recall*100:.2f}%)")
print(f"F1-Score:       {f1:.4f} ({f1*100:.2f}%)")
print(f"ROC-AUC:        {roc_auc:.4f} ({roc_auc*100:.2f}%)")
print(f"Specificity:    {specificity:.4f} ({specificity*100:.2f}%)")

print("\n🎯 CONFUSION MATRIX")
print("-" * 80)
print(f"True Negatives:  {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives:  {tp}")

print("\n📊 DETAILED CLASSIFICATION REPORT")
print("-" * 80)
print(classification_report(y_test, predictions, 
                          target_names=['Non-Outbreak', 'Outbreak'],
                          digits=4))

print("\n🏆 MODEL PERFORMANCE SUMMARY")
print("-" * 80)
print(f"Total Test Samples:     {len(y_test)}")
print(f"Correct Predictions:    {tp + tn}")
print(f"Incorrect Predictions:  {fp + fn}")
print(f"Error Rate:             {((fp + fn) / len(y_test))*100:.2f}%")

# Save detailed metrics
metrics_detailed = {
    "model_name": "Custom LSTM+Attention",
    "architecture": {
        "vocab_size": len(vocab),
        "embedding_dim": 256,
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.3
    },
    "metrics": {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "specificity": float(specificity)
    },
    "confusion_matrix": {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    },
    "test_set_size": int(len(y_test)),
    "training_params": {
        "test_size": 0.2,
        "random_state": 42,
        "max_sequence_length": 100
    }
}

output_file = 'results/custom_lstm_detailed_metrics.json'
with open(output_file, 'w') as f:
    json.dump(metrics_detailed, f, indent=2)

print(f"\n💾 Detailed metrics saved to: {output_file}")

print("\n" + "=" * 80)
print("✅ METRICS CALCULATION COMPLETE")
print("=" * 80)
