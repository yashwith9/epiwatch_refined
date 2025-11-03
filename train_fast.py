"""
Fast Training Pipeline for CPU
Trains models with optimized settings for faster training
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

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing.text_preprocessing import TextPreprocessor, DatasetBuilder
from models.custom_model import CustomEpiDetector, ModelTrainer, build_vocab, EpidemicDataset
from evaluation.model_evaluator import ModelEvaluator
from torch.utils.data import DataLoader

print("="*80)
print("EPIWATCH FAST TRAINING PIPELINE (CPU OPTIMIZED)")
print("="*80)

# Initialize
device = torch.device('cpu')
print(f"\n✓ Device: {device}")

preprocessor = TextPreprocessor()
dataset_builder = DatasetBuilder(preprocessor)
evaluator = ModelEvaluator()

# Load data
print("\n" + "="*80)
print("LOADING AND PREPROCESSING DATA")
print("="*80)

df = pd.read_csv('data/processed/epidemic_data.csv')
print(f"✓ Loaded {len(df)} samples\n")

# Preprocess
print("Preprocessing texts...")
df['processed_text'] = preprocessor.preprocess_dataset(df['text'].tolist(), show_progress=True)

# Balance and split
df_balanced = dataset_builder.balance_dataset(df, text_col='processed_text')
data_splits = dataset_builder.prepare_train_test_split(
    df_balanced,
    text_col='processed_text',
    test_size=0.2,
    val_size=0.1
)

print("\n✓ Data preparation complete!")

# Train Custom Model
print("\n" + "="*80)
print("TRAINING MODEL 1: CUSTOM NEURAL NETWORK")
print("="*80)

vocab = build_vocab(data_splits['train']['texts'], min_freq=2)
print(f"\nVocabulary size: {len(vocab)}")

train_dataset = EpidemicDataset(data_splits['train']['texts'], data_splits['train']['labels'], vocab, max_length=128)
val_dataset = EpidemicDataset(data_splits['val']['texts'], data_splits['val']['labels'], vocab, max_length=128)
test_dataset = EpidemicDataset(data_splits['test']['texts'], data_splits['test']['labels'], vocab, max_length=128)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

model = CustomEpiDetector(
    vocab_size=len(vocab),
    embedding_dim=128,
    hidden_dim=64,
    num_layers=1,
    dropout=0.3
)

print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")

trainer = ModelTrainer(model, device=device)
trainer.train(train_loader, val_loader, epochs=5, lr=0.001)

# Evaluate
predictions, _ = trainer.predict(test_loader)
y_pred = (predictions > 0.5).astype(int)
y_true = np.array(data_splits['test']['labels'])

evaluator.evaluate_model(
    "Custom Neural Network",
    y_true,
    y_pred,
    y_prob=predictions,
    inference_time=0.02,
    model_size=15
)

print("\n✓ Custom model training complete!")

# Train DistilBERT (fastest transformer)
print("\n" + "="*80)
print("TRAINING MODEL 2: DISTILBERT (EFFICIENT)")
print("="*80)

try:
    from models.pretrained_models import PretrainedEpiDetector
    
    model_dist = PretrainedEpiDetector('distilbert-base-multilingual-cased', device=device)
    
    train_loader_bert = model_dist.prepare_dataloader(
        data_splits['train']['texts'][:1000],  # Reduced for speed
        data_splits['train']['labels'][:1000],
        batch_size=8,
        max_length=128
    )
    
    val_loader_bert = model_dist.prepare_dataloader(
        data_splits['val']['texts'],
        data_splits['val']['labels'],
        batch_size=8,
        max_length=128,
        shuffle=False
    )
    
    model_dist.train(train_loader_bert, val_loader_bert, epochs=2, learning_rate=3e-5)
    
    predictions_dist, probabilities_dist = model_dist.predict(
        data_splits['test']['texts'],
        batch_size=8,
        max_length=128
    )
    
    timing = model_dist.measure_inference_time(data_splits['test']['texts'][0], num_runs=10)
    
    evaluator.evaluate_model(
        "DistilBERT-Multilingual",
        y_true,
        predictions_dist,
        y_prob=probabilities_dist,
        inference_time=timing['mean'],
        model_size=270
    )
    
    print("\n✓ DistilBERT training complete!")
    
except Exception as e:
    print(f"\n❌ Error training DistilBERT: {str(e)}")
    print("Continuing with custom model only...")

# Simple baseline models for comparison
print("\n" + "="*80)
print("TRAINING MODEL 3-5: BASELINE MODELS")
print("="*80)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Prepare text data
X_train = data_splits['train']['texts']
y_train = data_splits['train']['labels']
X_test = data_splits['test']['texts']

# TF-IDF vectorization
print("\nVectorizing texts...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Naive Bayes
print("\nTraining Naive Bayes...")
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
y_pred_nb = nb_model.predict(X_test_tfidf)
y_prob_nb = nb_model.predict_proba(X_test_tfidf)[:, 1]

evaluator.evaluate_model(
    "Naive Bayes (TF-IDF)",
    y_true,
    y_pred_nb,
    y_prob=y_prob_nb,
    inference_time=0.001,
    model_size=5
)

# Train Logistic Regression
print("Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_tfidf, y_train)
y_pred_lr = lr_model.predict(X_test_tfidf)
y_prob_lr = lr_model.predict_proba(X_test_tfidf)[:, 1]

evaluator.evaluate_model(
    "Logistic Regression (TF-IDF)",
    y_true,
    y_pred_lr,
    y_prob=y_prob_lr,
    inference_time=0.001,
    model_size=10
)

# Train Random Forest
print("Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_tfidf, y_train)
y_pred_rf = rf_model.predict(X_test_tfidf)
y_prob_rf = rf_model.predict_proba(X_test_tfidf)[:, 1]

evaluator.evaluate_model(
    "Random Forest (TF-IDF)",
    y_true,
    y_pred_rf,
    y_prob=y_prob_rf,
    inference_time=0.005,
    model_size=50
)

print("\n✓ All baseline models trained!")

# Compare and visualize
print("\n" + "="*80)
print("MODEL COMPARISON AND RESULTS")
print("="*80)

evaluator.print_summary()

# Save results
os.makedirs('outputs/visualizations', exist_ok=True)
evaluator.plot_comparison(save_path='outputs/visualizations/model_comparison.png')
evaluator.plot_confusion_matrices(save_path='outputs/visualizations/confusion_matrices.png')
evaluator.save_results(filepath='outputs/model_comparison_results.json')

recommendation = evaluator.generate_recommendation()
with open('outputs/recommendation.json', 'w') as f:
    json.dump(recommendation, f, indent=4)

comparison_df = evaluator.get_comparison_table()
comparison_df.to_csv('outputs/model_comparison_table.csv', index=False)

print("\n" + "🎉 " * 40)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("🎉 " * 40)

print(f"\n✓ All results saved to outputs/")
print(f"✓ Best model: {recommendation['recommended_model']}")
print(f"✓ Visualizations: outputs/visualizations/")

print("\n" + "="*80)
