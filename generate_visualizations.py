"""
Generate comprehensive visualizations for model performance
Including: ROC curves, accuracy comparison, training time, inference speed, confusion matrix
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

sys.path.insert(0, 'c:\\Users\\Bruger\\OneDrive\\Desktop\\NLP')
from src.models.custom_model import CustomEpiDetector
from src.preprocessing.text_preprocessing import TextPreprocessor

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

print("=" * 80)
print("📊 GENERATING MODEL PERFORMANCE VISUALIZATIONS")
print("=" * 80)

# Load results
print("\n📁 Loading model results...")
with open('results/ultrafast_results.json', 'r') as f:
    results = json.load(f)

with open('results/custom_lstm_detailed_metrics.json', 'r') as f:
    lstm_detailed = json.load(f)

print("✓ Results loaded successfully")

# ============================================================================
# 1. ROC CURVE FOR CUSTOM LSTM
# ============================================================================
print("\n📈 Generating ROC Curve for Custom LSTM+Attention...")

# Load data and generate predictions
data_path = 'data/processed/epidemic_data.csv'
df = pd.read_csv(data_path)
X = df['text'].values
y = df['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Load vocabulary and preprocess
with open('models/saved/vocab.json', 'r', encoding='utf-8') as f:
    vocab = json.load(f)

preprocessor = TextPreprocessor()

def text_to_indices(text, vocab, max_len=100):
    tokens = preprocessor.preprocess(text).split()
    indices = [vocab.get(token, vocab.get('<UNK>', 0)) for token in tokens]
    if len(indices) < max_len:
        indices += [vocab.get('<PAD>', 0)] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    return indices

X_test_indices = [text_to_indices(text, vocab) for text in X_test]
X_test_tensor = torch.LongTensor(X_test_indices)

# Load model and get predictions
device = torch.device('cpu')
model = CustomEpiDetector(vocab_size=len(vocab), embedding_dim=256, hidden_dim=128, num_layers=2, dropout=0.3)
checkpoint = torch.load('models/saved/custom_best.pt', map_location=device)
model.load_state_dict(checkpoint)
model.eval()

with torch.no_grad():
    outputs, _ = model(X_test_tensor)
    y_scores = outputs.squeeze().numpy()

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Custom LSTM+Attention (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier (AUC = 0.50)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Custom LSTM+Attention Model', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/roc_curve_lstm.png', dpi=300, bbox_inches='tight')
print("✓ ROC curve saved: results/roc_curve_lstm.png")
plt.close()

# ============================================================================
# 2. ACCURACY COMPARISON BAR CHART
# ============================================================================
print("\n📊 Generating Accuracy Comparison Chart...")

models = list(results.keys())
accuracies = [results[m]['accuracy'] * 100 for m in models]

plt.figure(figsize=(12, 7))
colors = ['#FF6B6B' if acc < 60 else '#FFD93D' if acc < 90 else '#6BCB77' for acc in accuracies]
bars = plt.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.xlabel('Models', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
plt.ylim(0, 110)
plt.xticks(rotation=45, ha='right')
plt.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='90% Threshold')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('results/accuracy_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Accuracy chart saved: results/accuracy_comparison.png")
plt.close()

# ============================================================================
# 3. F1-SCORE COMPARISON
# ============================================================================
print("\n📊 Generating F1-Score Comparison Chart...")

f1_scores = [results[m]['f1'] * 100 if results[m]['f1'] else 0 for m in models]

plt.figure(figsize=(12, 7))
colors = ['#FF6B6B' if f1 < 60 else '#FFD93D' if f1 < 90 else '#6BCB77' for f1 in f1_scores]
bars = plt.bar(models, f1_scores, color=colors, edgecolor='black', linewidth=1.5)

for bar, f1 in zip(bars, f1_scores):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{f1:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.xlabel('Models', fontsize=12, fontweight='bold')
plt.ylabel('F1-Score (%)', fontsize=12, fontweight='bold')
plt.title('Model F1-Score Comparison', fontsize=14, fontweight='bold')
plt.ylim(0, 110)
plt.xticks(rotation=45, ha='right')
plt.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='90% Threshold')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('results/f1_score_comparison.png', dpi=300, bbox_inches='tight')
print("✓ F1-score chart saved: results/f1_score_comparison.png")
plt.close()

# ============================================================================
# 4. TRAINING TIME COMPARISON
# ============================================================================
print("\n⏱️  Generating Training Time Comparison...")

train_times = [results[m]['train_time'] for m in models]

plt.figure(figsize=(12, 7))
colors = ['#4ECDC4' if t < 100 else '#FF6B6B' if t > 400 else '#FFD93D' for t in train_times]
bars = plt.barh(models, train_times, color=colors, edgecolor='black', linewidth=1.5)

for bar, time in zip(bars, train_times):
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height()/2.,
             f'{time:.1f}s', ha='left', va='center', fontsize=11, fontweight='bold', 
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.xlabel('Training Time (seconds)', fontsize=12, fontweight='bold')
plt.ylabel('Models', fontsize=12, fontweight='bold')
plt.title('Model Training Time Comparison', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('results/training_time_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Training time chart saved: results/training_time_comparison.png")
plt.close()

# ============================================================================
# 5. INFERENCE SPEED COMPARISON
# ============================================================================
print("\n⚡ Generating Inference Speed Comparison...")

inference_times = [results[m]['inference_ms'] for m in models]

plt.figure(figsize=(12, 7))
colors = ['#6BCB77' if t < 100 else '#FFD93D' if t < 500 else '#FF6B6B' for t in inference_times]
bars = plt.barh(models, inference_times, color=colors, edgecolor='black', linewidth=1.5)

for bar, time in zip(bars, inference_times):
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height()/2.,
             f'{time:.1f}ms', ha='left', va='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.xlabel('Inference Time (milliseconds)', fontsize=12, fontweight='bold')
plt.ylabel('Models', fontsize=12, fontweight='bold')
plt.title('Model Inference Speed Comparison (Lower is Better)', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('results/inference_speed_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Inference speed chart saved: results/inference_speed_comparison.png")
plt.close()

# ============================================================================
# 6. CONFUSION MATRIX FOR CUSTOM LSTM
# ============================================================================
print("\n🎯 Generating Confusion Matrix...")

cm = np.array([[lstm_detailed['confusion_matrix']['true_negatives'], 
                lstm_detailed['confusion_matrix']['false_positives']],
               [lstm_detailed['confusion_matrix']['false_negatives'], 
                lstm_detailed['confusion_matrix']['true_positives']]])

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True, 
            xticklabels=['Non-Outbreak', 'Outbreak'],
            yticklabels=['Non-Outbreak', 'Outbreak'],
            cbar_kws={'label': 'Count'},
            annot_kws={'size': 16, 'weight': 'bold'})
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.title('Confusion Matrix - Custom LSTM+Attention', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/confusion_matrix_lstm.png', dpi=300, bbox_inches='tight')
print("✓ Confusion matrix saved: results/confusion_matrix_lstm.png")
plt.close()

# ============================================================================
# 7. MULTI-METRIC RADAR CHART
# ============================================================================
print("\n🕸️  Generating Multi-Metric Radar Chart...")

# Prepare data for Custom LSTM
categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
lstm_values = [
    lstm_detailed['metrics']['accuracy'] * 100,
    lstm_detailed['metrics']['precision'] * 100,
    lstm_detailed['metrics']['recall'] * 100,
    lstm_detailed['metrics']['f1_score'] * 100,
    lstm_detailed['metrics']['roc_auc'] * 100
]

# Number of variables
num_vars = len(categories)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
lstm_values += lstm_values[:1]  # Complete the circle
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
ax.plot(angles, lstm_values, 'o-', linewidth=2, label='Custom LSTM+Attention', color='#FF6B6B')
ax.fill(angles, lstm_values, alpha=0.25, color='#FF6B6B')
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=12, fontweight='bold')
ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], size=10)
ax.grid(True)
ax.set_title('Custom LSTM+Attention - Multi-Metric Performance', 
             size=14, fontweight='bold', pad=20)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig('results/radar_chart_lstm.png', dpi=300, bbox_inches='tight')
print("✓ Radar chart saved: results/radar_chart_lstm.png")
plt.close()

# ============================================================================
# 8. OVERALL PERFORMANCE HEATMAP
# ============================================================================
print("\n🔥 Generating Overall Performance Heatmap...")

# Create performance matrix
performance_data = []
for model in models:
    acc = results[model]['accuracy'] * 100 if results[model]['accuracy'] else 0
    f1 = results[model]['f1'] * 100 if results[model]['f1'] else 0
    speed_score = 100 - (results[model]['inference_ms'] / max(inference_times) * 100)
    train_score = 100 - (results[model]['train_time'] / max(train_times) * 100)
    performance_data.append([acc, f1, speed_score, train_score])

performance_df = pd.DataFrame(performance_data, 
                             columns=['Accuracy', 'F1-Score', 'Inference Speed', 'Training Speed'],
                             index=models)

plt.figure(figsize=(12, 8))
sns.heatmap(performance_df, annot=True, fmt='.1f', cmap='RdYlGn', 
            cbar_kws={'label': 'Score (%)'},
            linewidths=0.5, linecolor='gray')
plt.title('Overall Model Performance Heatmap', fontsize=14, fontweight='bold')
plt.xlabel('Metrics', fontsize=12, fontweight='bold')
plt.ylabel('Models', fontsize=12, fontweight='bold')
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('results/performance_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Performance heatmap saved: results/performance_heatmap.png")
plt.close()

# ============================================================================
# 9. COMBINED METRICS COMPARISON
# ============================================================================
print("\n📊 Generating Combined Metrics Comparison...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Comprehensive Model Performance Comparison', fontsize=16, fontweight='bold')

# Subplot 1: Accuracy
ax1 = axes[0, 0]
ax1.bar(models, accuracies, color='#6BCB77', edgecolor='black', linewidth=1.5)
ax1.set_title('Accuracy', fontsize=12, fontweight='bold')
ax1.set_ylabel('Percentage (%)')
ax1.set_ylim(0, 110)
ax1.tick_params(axis='x', rotation=45)
ax1.grid(axis='y', alpha=0.3)

# Subplot 2: F1-Score
ax2 = axes[0, 1]
ax2.bar(models, f1_scores, color='#4ECDC4', edgecolor='black', linewidth=1.5)
ax2.set_title('F1-Score', fontsize=12, fontweight='bold')
ax2.set_ylabel('Percentage (%)')
ax2.set_ylim(0, 110)
ax2.tick_params(axis='x', rotation=45)
ax2.grid(axis='y', alpha=0.3)

# Subplot 3: Training Time
ax3 = axes[1, 0]
ax3.barh(models, train_times, color='#FFD93D', edgecolor='black', linewidth=1.5)
ax3.set_title('Training Time', fontsize=12, fontweight='bold')
ax3.set_xlabel('Seconds')
ax3.grid(axis='x', alpha=0.3)

# Subplot 4: Inference Speed
ax4 = axes[1, 1]
ax4.barh(models, inference_times, color='#FF6B6B', edgecolor='black', linewidth=1.5)
ax4.set_title('Inference Speed', fontsize=12, fontweight='bold')
ax4.set_xlabel('Milliseconds')
ax4.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('results/combined_metrics_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Combined metrics chart saved: results/combined_metrics_comparison.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
print("=" * 80)
print("\n📁 Generated Files:")
print("   1. results/roc_curve_lstm.png - ROC Curve for Custom LSTM")
print("   2. results/accuracy_comparison.png - Accuracy comparison across models")
print("   3. results/f1_score_comparison.png - F1-Score comparison")
print("   4. results/training_time_comparison.png - Training time comparison")
print("   5. results/inference_speed_comparison.png - Inference speed comparison")
print("   6. results/confusion_matrix_lstm.png - Confusion matrix")
print("   7. results/radar_chart_lstm.png - Multi-metric radar chart")
print("   8. results/performance_heatmap.png - Overall performance heatmap")
print("   9. results/combined_metrics_comparison.png - Combined comparison")
print("\n" + "=" * 80)
