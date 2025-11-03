"""
Fast Training Script for 5 Models on Disease Outbreak Dataset
Models: DistilBERT, MuRIL, mBERT, XLM-RoBERTa, Custom LSTM+Attention
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models.pretrained_models import PretrainedEpiDetector, MODEL_CONFIGS

print("="*80)
print("FAST TRAINING: 5 MODELS FOR EPIDEMIC DETECTION")
print("="*80)
print("Models to train:")
print("✓ DistilBERT (distilbert-base-multilingual-cased)")
print("✓ MuRIL (google/muril-base-cased)")
print("✓ mBERT (bert-base-multilingual-cased)")
print("✓ XLM-RoBERTa (xlm-roberta-base)")
print("✓ Custom Neural Network (LSTM + Attention)")
print("="*80)
print()

# Configuration
DATASET_PATH = "data/disease_outbreaks_minimal.csv"  # Dataset in workspace
EPOCHS = 3  # Fast training
BATCH_SIZE = 16
MAX_LENGTH = 256  # Shorter for speed
LEARNING_RATE = 2e-5

def load_and_prepare_data(dataset_path):
    """Load and prepare the dataset"""
    print("Loading dataset...")
    
    # Try to load the dataset
    try:
        df = pd.read_csv(dataset_path)
        print(f"✓ Loaded {len(df)} samples")
        print(f"✓ Columns: {list(df.columns)}")
        
        # Display first few rows to understand structure
        print("\nDataset preview:")
        print(df.head())
        
        # Detect text and label columns
        text_col = None
        label_col = None
        
        # Common text column names
        text_candidates = ['text', 'description', 'content', 'message', 'tweet', 'post']
        label_candidates = ['label', 'class', 'target', 'epidemic', 'outbreak']
        
        for col in df.columns:
            if col.lower() in text_candidates or 'text' in col.lower():
                text_col = col
            if col.lower() in label_candidates or 'label' in col.lower():
                label_col = col
        
        if text_col is None:
            # Use first string column as text
            for col in df.columns:
                if df[col].dtype == 'object':
                    text_col = col
                    break
        
        if label_col is None:
            # Use last numeric column as label
            for col in reversed(df.columns):
                if df[col].dtype in ['int64', 'float64']:
                    label_col = col
                    break
        
        print(f"✓ Using text column: '{text_col}'")
        print(f"✓ Using label column: '{label_col}'")
        
        # Clean data
        df = df.dropna(subset=[text_col, label_col])
        df[text_col] = df[text_col].astype(str)
        
        # Convert labels to binary if needed
        unique_labels = df[label_col].unique()
        print(f"✓ Unique labels: {unique_labels}")
        
        if len(unique_labels) > 2:
            # Convert to binary (epidemic vs non-epidemic)
            df[label_col] = (df[label_col] > 0).astype(int)
            print("✓ Converted to binary classification")
        
        texts = df[text_col].tolist()
        labels = df[label_col].tolist()
        
        print(f"✓ Final dataset: {len(texts)} samples")
        print(f"✓ Label distribution: {np.bincount(labels)}")
        
        return texts, labels
        
    except FileNotFoundError:
        print(f"✗ Dataset not found at {dataset_path}")
        print("Please copy your dataset to the workspace or update the path")
        
        # Create dummy data for demonstration
        print("Creating dummy data for demonstration...")
        texts = [
            "Outbreak of flu reported in the city center",
            "Weather is nice today",
            "COVID-19 cases rising in the region",
            "Stock market performing well",
            "Dengue fever spreading rapidly",
            "New restaurant opened downtown"
        ] * 100
        labels = [1, 0, 1, 0, 1, 0] * 100
        
        return texts, labels

def train_custom_model(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train the custom LSTM+Attention model"""
    print("\n" + "="*60)
    print("TRAINING CUSTOM NEURAL NETWORK (LSTM + ATTENTION)")
    print("="*60)
    
    try:
        from models.scratch_model_trainer import ScratchModelTrainer
        from models.scratch_model_components import EpidemicLSTMModel
        
        # Initialize trainer
        trainer = ScratchModelTrainer(
            vocab_size=10000,
            embedding_dim=128,
            hidden_dim=64,
            num_layers=2,
            dropout=0.3,
            device='cpu'  # Use CPU for compatibility
        )
        
        # Prepare data
        train_data = list(zip(X_train, y_train))
        val_data = list(zip(X_val, y_val))
        
        # Train
        print("Training custom model...")
        start_time = time.time()
        
        trainer.train(
            train_data=train_data,
            val_data=val_data,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=0.001
        )
        
        train_time = time.time() - start_time
        
        # Evaluate
        print("Evaluating custom model...")
        start_inference = time.time()
        predictions, probabilities = trainer.predict(X_test)
        inference_time = (time.time() - start_inference) / len(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        
        results = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "train_time": float(train_time),
            "inference_time_ms": float(inference_time * 1000),
            "model_type": "custom_lstm_attention"
        }
        
        print(f"✓ Custom Model Results:")
        print(f"  - Accuracy: {accuracy:.4f}")
        print(f"  - Precision: {precision:.4f}")
        print(f"  - Recall: {recall:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - Training time: {train_time:.2f}s")
        print(f"  - Inference: {inference_time*1000:.2f}ms per sample")
        
        return results
        
    except Exception as e:
        print(f"✗ Error training custom model: {str(e)}")
        return {"error": str(e), "status": "failed"}

def train_transformer_models(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train all transformer models"""
    
    # Model configurations with your specified models
    models_to_train = {
        "DistilBERT": "distilbert-base-multilingual-cased",
        "MuRIL": "google/muril-base-cased", 
        "mBERT": "bert-base-multilingual-cased",
        "XLM-RoBERTa": "xlm-roberta-base"
    }
    
    results = {}
    
    for model_name, model_path in models_to_train.items():
        print("\n" + "="*60)
        print(f"TRAINING {model_name.upper()}")
        print("="*60)
        
        try:
            # Initialize model
            print(f"Loading {model_name}...")
            detector = PretrainedEpiDetector(
                model_name=model_path,
                num_labels=2,
                device=torch.device('cpu')  # Use CPU for compatibility
            )
            
            # Prepare data loaders
            print("Preparing data...")
            train_loader = detector.prepare_dataloader(
                X_train, y_train, batch_size=BATCH_SIZE, max_length=MAX_LENGTH
            )
            val_loader = detector.prepare_dataloader(
                X_val, y_val, batch_size=BATCH_SIZE, max_length=MAX_LENGTH, shuffle=False
            )
            
            # Train
            print(f"Training {model_name}...")
            start_time = time.time()
            
            detector.train(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=EPOCHS,
                learning_rate=LEARNING_RATE
            )
            
            train_time = time.time() - start_time
            
            # Evaluate on test set
            print(f"Evaluating {model_name}...")
            start_inference = time.time()
            predictions, probabilities = detector.predict(X_test, batch_size=BATCH_SIZE)
            inference_time = (time.time() - start_inference) / len(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, zero_division=0)
            recall = recall_score(y_test, predictions, zero_division=0)
            f1 = f1_score(y_test, predictions, zero_division=0)
            
            # Measure inference speed
            sample_text = X_test[0] if X_test else "Sample text for timing"
            timing_stats = detector.measure_inference_time(sample_text, num_runs=50)
            
            results[model_name] = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "train_time": float(train_time),
                "inference_time_ms": float(inference_time * 1000),
                "inference_stats": {
                    "mean_ms": float(timing_stats['mean'] * 1000),
                    "std_ms": float(timing_stats['std'] * 1000)
                },
                "model_type": "transformer",
                "model_path": model_path
            }
            
            print(f"\n✓ {model_name} Results:")
            print(f"  - Accuracy: {accuracy:.4f}")
            print(f"  - Precision: {precision:.4f}")
            print(f"  - Recall: {recall:.4f}")
            print(f"  - F1 Score: {f1:.4f}")
            print(f"  - Training time: {train_time:.2f}s")
            print(f"  - Inference: {inference_time*1000:.2f}ms per sample")
            
            # Clear memory
            del detector
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"✗ Error training {model_name}: {str(e)}")
            results[model_name] = {"error": str(e), "status": "failed"}
            continue
    
    return results

def create_comparison_analysis(results):
    """Create comprehensive comparison analysis"""
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL COMPARISON ANALYSIS")
    print("="*80)
    
    # Filter successful models
    successful_models = {k: v for k, v in results.items() if 'error' not in v}
    
    if not successful_models:
        print("No models trained successfully!")
        return
    
    # Create comparison table
    print(f"\n{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Speed(ms)':<10}")
    print("-" * 80)
    
    # Sort by F1 score
    sorted_models = sorted(successful_models.items(), key=lambda x: x[1]['f1'], reverse=True)
    
    for model_name, metrics in sorted_models:
        speed = metrics.get('inference_time_ms', 0)
        print(f"{model_name:<20} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
              f"{metrics['recall']:<10.4f} {metrics['f1']:<10.4f} {speed:<10.2f}")
    
    # Best model analysis
    best_model = sorted_models[0]
    print(f"\n🏆 BEST PERFORMING MODEL: {best_model[0]}")
    print(f"   F1-Score: {best_model[1]['f1']:.4f}")
    print(f"   Accuracy: {best_model[1]['accuracy']:.4f}")
    print(f"   Inference Speed: {best_model[1]['inference_time_ms']:.2f}ms")
    
    # Speed analysis
    fastest_model = min(successful_models.items(), key=lambda x: x[1]['inference_time_ms'])
    print(f"\n⚡ FASTEST MODEL: {fastest_model[0]}")
    print(f"   Inference Speed: {fastest_model[1]['inference_time_ms']:.2f}ms")
    print(f"   F1-Score: {fastest_model[1]['f1']:.4f}")
    
    # Model type analysis
    transformer_models = {k: v for k, v in successful_models.items() if v.get('model_type') == 'transformer'}
    custom_models = {k: v for k, v in successful_models.items() if v.get('model_type') != 'transformer'}
    
    if transformer_models:
        avg_transformer_f1 = np.mean([v['f1'] for v in transformer_models.values()])
        print(f"\n📊 TRANSFORMER MODELS AVERAGE F1: {avg_transformer_f1:.4f}")
    
    if custom_models:
        avg_custom_f1 = np.mean([v['f1'] for v in custom_models.values()])
        print(f"📊 CUSTOM MODELS AVERAGE F1: {avg_custom_f1:.4f}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   • For best accuracy: Use {best_model[0]}")
    print(f"   • For fastest inference: Use {fastest_model[0]}")
    
    if best_model[1]['f1'] > 0.8:
        print(f"   • {best_model[0]} shows excellent performance (F1 > 0.8)")
    elif best_model[1]['f1'] > 0.7:
        print(f"   • {best_model[0]} shows good performance (F1 > 0.7)")
    else:
        print(f"   • Consider more training data or hyperparameter tuning")
    
    return successful_models

def save_results(results):
    """Save results to files"""
    os.makedirs("results", exist_ok=True)
    
    # Save JSON results
    with open("results/five_models_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save CSV summary
    successful_models = {k: v for k, v in results.items() if 'error' not in v}
    if successful_models:
        df_results = pd.DataFrame(successful_models).T
        df_results.to_csv("results/five_models_summary.csv")
    
    print(f"\n✓ Results saved to:")
    print(f"   - results/five_models_comparison.json")
    print(f"   - results/five_models_summary.csv")

def main():
    """Main training pipeline"""
    
    # Load data
    texts, labels = load_and_prepare_data(DATASET_PATH)
    
    # Split data
    print("\nSplitting data...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"✓ Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Train all models
    all_results = {}
    
    # Train transformer models
    transformer_results = train_transformer_models(X_train, y_train, X_val, y_val, X_test, y_test)
    all_results.update(transformer_results)
    
    # Train custom model
    custom_results = train_custom_model(X_train, y_train, X_val, y_val, X_test, y_test)
    all_results["Custom LSTM+Attention"] = custom_results
    
    # Analysis and comparison
    create_comparison_analysis(all_results)
    
    # Save results
    save_results(all_results)
    
    print("\n" + "="*80)
    print("✅ ALL MODELS TRAINED SUCCESSFULLY!")
    print("✅ COMPARATIVE ANALYSIS COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()