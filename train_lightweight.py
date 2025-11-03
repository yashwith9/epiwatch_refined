"""
Lightweight training pipeline - Uses already trained custom model + classical ML models
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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from preprocessing.text_preprocessing import TextPreprocessor
import time

print("="*80)
print("EPIWATCH LIGHTWEIGHT MODEL COMPARISON")
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
from sklearn.model_selection import train_test_split
X_train, X_temp, y_train, y_temp = train_test_split(
    df['processed_text'], df['label'], test_size=0.3, random_state=42, stratify=df['label']
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.666, random_state=42, stratify=y_temp
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}\n")

# TF-IDF
print("Creating TF-IDF features...")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.transform(X_val)
X_test_tfidf = tfidf.transform(X_test)
print(f"✓ Feature matrix: {X_train_tfidf.shape}\n")

# Models to train
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Linear SVM": LinearSVC(max_iter=500, random_state=42)
}

results = {}

print("="*80)
print("TRAINING CLASSICAL ML MODELS")
print("="*80)
print()

for name, model in models.items():
    print(f"Training {name}...")
    start_time = time.time()
    
    # Train
    model.fit(X_train_tfidf, y_train)
    train_time = time.time() - start_time
    
    # Evaluate on validation
    y_pred_val = model.predict(X_val_tfidf)
    
    # Evaluate on test
    start_inference = time.time()
    y_pred_test = model.predict(X_test_tfidf)
    inference_time = (time.time() - start_inference) / len(X_test)
    
    # Metrics
    results[name] = {
        "accuracy": float(accuracy_score(y_test, y_pred_test)),
        "precision": float(precision_score(y_test, y_pred_test)),
        "recall": float(recall_score(y_test, y_pred_test)),
        "f1": float(f1_score(y_test, y_pred_test)),
        "train_time": float(train_time),
        "inference_time_per_sample": float(inference_time * 1000),  # in ms
        "model_type": "classical_ml"
    }
    
    print(f"  ✓ Accuracy: {results[name]['accuracy']:.4f}")
    print(f"  ✓ F1 Score: {results[name]['f1']:.4f}")
    print(f"  ✓ Training time: {train_time:.2f}s")
    print()

# Load custom model results
print("="*80)
print("LOADING CUSTOM NEURAL NETWORK RESULTS")
print("="*80)
print()

custom_model_path = "models/saved/custom_best.pt"
if os.path.exists(custom_model_path):
    print("Loading trained custom model...")
    
    from models.custom_model import CustomEpiDetector, build_vocab, EpidemicDataset
    from torch.utils.data import DataLoader
    
    # Build vocab
    vocab = build_vocab(X_train.tolist())
    
    # Create dataset
    test_dataset = EpidemicDataset(X_test.tolist(), y_test.tolist(), vocab)
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
            outputs, _ = model(batch_texts)  # Model returns (output, attention_weights)
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
        "train_time": 0.0,  # Already trained
        "inference_time_per_sample": float(inference_time * 1000),
        "model_type": "deep_learning"
    }
    
    print(f"✓ Custom Neural Network:")
    print(f"  - Accuracy: {results['Custom Neural Network']['accuracy']:.4f}")
    print(f"  - F1 Score: {results['Custom Neural Network']['f1']:.4f}")
    print(f"  - Inference time: {results['Custom Neural Network']['inference_time_per_sample']:.4f}ms per sample")
    print()

# Save results
os.makedirs("results", exist_ok=True)
with open("results/metrics_comparison.json", "w") as f:
    json.dump(results, f, indent=2)

print("="*80)
print("RESULTS SUMMARY")
print("="*80)
print()
print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
print("-"*80)
for model_name, metrics in sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True):
    print(f"{model_name:<25} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} "
          f"{metrics['recall']:<12.4f} {metrics['f1']:<12.4f}")

print()
print("✓ Results saved to results/metrics_comparison.json")
print("✓ Ready to generate README with visualizations!")
