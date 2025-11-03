"""
Model Evaluation and Comparison Framework
Comprehensive evaluation of all 5 models
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
import torch
from datetime import datetime


class ModelEvaluator:
    """
    Comprehensive model evaluation and comparison
    """
    
    def __init__(self):
        self.results = {}
        
    def evaluate_model(self, model_name, y_true, y_pred, y_prob=None, 
                      inference_time=None, model_size=None):
        """
        Evaluate a single model with comprehensive metrics
        
        Args:
            model_name: Name of the model
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)
            inference_time: Average inference time in seconds
            model_size: Model size in MB
        """
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary')
        recall = recall_score(y_true, y_pred, average='binary')
        f1 = f1_score(y_true, y_pred, average='binary')
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Specificity (True Negative Rate)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # AUC-ROC
        auc = None
        if y_prob is not None:
            try:
                auc = roc_auc_score(y_true, y_prob)
            except:
                auc = None
        
        # Store results
        self.results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'auc_roc': auc,
            'confusion_matrix': cm,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'inference_time_ms': inference_time * 1000 if inference_time else None,
            'model_size_mb': model_size,
            'total_samples': len(y_true),
            'timestamp': datetime.now().isoformat()
        }
        
        return self.results[model_name]
    
    def get_comparison_table(self):
        """Generate comparison table across all models"""
        
        if not self.results:
            return None
        
        comparison_data = []
        
        for model_name, metrics in self.results.items():
            row = {
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'Specificity': f"{metrics['specificity']:.4f}",
                'AUC-ROC': f"{metrics['auc_roc']:.4f}" if metrics['auc_roc'] else 'N/A',
                'Inference Time (ms)': f"{metrics['inference_time_ms']:.2f}" if metrics['inference_time_ms'] else 'N/A',
                'Model Size (MB)': f"{metrics['model_size_mb']:.2f}" if metrics['model_size_mb'] else 'N/A'
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        return df
    
    def plot_comparison(self, save_path='outputs/visualizations/model_comparison.png'):
        """Create comprehensive comparison visualizations"""
        
        if not self.results:
            print("No results to plot!")
            return
        
        # Prepare data
        models = list(self.results.keys())
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive Model Comparison - EpiWatch', fontsize=16, fontweight='bold')
        
        # 1. Performance Metrics Comparison
        ax1 = axes[0, 0]
        data = {metric: [self.results[m][metric] for m in models] 
                for metric in metrics_to_plot}
        
        x = np.arange(len(models))
        width = 0.15
        
        for i, metric in enumerate(metrics_to_plot):
            ax1.bar(x + i*width, data[metric], width, label=metric.replace('_', ' ').title())
        
        ax1.set_xlabel('Models', fontweight='bold')
        ax1.set_ylabel('Score', fontweight='bold')
        ax1.set_title('Performance Metrics Comparison')
        ax1.set_xticks(x + width * 2)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim([0, 1.1])
        
        # 2. F1-Score Ranking
        ax2 = axes[0, 1]
        f1_scores = [(m, self.results[m]['f1_score']) for m in models]
        f1_scores.sort(key=lambda x: x[1], reverse=True)
        
        colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(f1_scores))]
        ax2.barh([x[0] for x in f1_scores], [x[1] for x in f1_scores], color=colors)
        ax2.set_xlabel('F1-Score', fontweight='bold')
        ax2.set_title('Model Ranking by F1-Score')
        ax2.set_xlim([0, 1.1])
        
        for i, (model, score) in enumerate(f1_scores):
            ax2.text(score + 0.01, i, f'{score:.4f}', va='center')
        
        # 3. Inference Time Comparison
        ax3 = axes[0, 2]
        inference_times = [(m, self.results[m]['inference_time_ms']) 
                          for m in models if self.results[m]['inference_time_ms']]
        
        if inference_times:
            ax3.bar([x[0] for x in inference_times], [x[1] for x in inference_times], 
                   color='#e74c3c')
            ax3.set_ylabel('Inference Time (ms)', fontweight='bold')
            ax3.set_title('Inference Speed Comparison')
            ax3.set_xticklabels([x[0] for x in inference_times], rotation=45, ha='right')
            ax3.grid(axis='y', alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No timing data available', 
                    ha='center', va='center', transform=ax3.transAxes)
        
        # 4. Confusion Matrix for Best Model
        ax4 = axes[1, 0]
        best_model = max(models, key=lambda m: self.results[m]['f1_score'])
        cm = self.results[best_model]['confusion_matrix']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4, 
                   cbar_kws={'label': 'Count'})
        ax4.set_xlabel('Predicted Label', fontweight='bold')
        ax4.set_ylabel('True Label', fontweight='bold')
        ax4.set_title(f'Confusion Matrix - {best_model} (Best Model)')
        ax4.set_xticklabels(['Non-Outbreak', 'Outbreak'])
        ax4.set_yticklabels(['Non-Outbreak', 'Outbreak'])
        
        # 5. Precision-Recall Trade-off
        ax5 = axes[1, 1]
        precisions = [self.results[m]['precision'] for m in models]
        recalls = [self.results[m]['recall'] for m in models]
        
        ax5.scatter(recalls, precisions, s=200, alpha=0.6, c=range(len(models)), cmap='viridis')
        
        for i, model in enumerate(models):
            ax5.annotate(model, (recalls[i], precisions[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax5.set_xlabel('Recall', fontweight='bold')
        ax5.set_ylabel('Precision', fontweight='bold')
        ax5.set_title('Precision-Recall Trade-off')
        ax5.grid(alpha=0.3)
        ax5.set_xlim([0, 1.1])
        ax5.set_ylim([0, 1.1])
        
        # 6. Model Performance Score (Weighted)
        ax6 = axes[1, 2]
        # Weighted score: 40% F1 + 30% Precision + 20% Recall + 10% Specificity
        scores = []
        for m in models:
            score = (0.4 * self.results[m]['f1_score'] + 
                    0.3 * self.results[m]['precision'] + 
                    0.2 * self.results[m]['recall'] + 
                    0.1 * self.results[m]['specificity'])
            scores.append((m, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        colors = plt.cm.RdYlGn([x[1] for x in scores])
        
        ax6.barh([x[0] for x in scores], [x[1] for x in scores], color=colors)
        ax6.set_xlabel('Weighted Performance Score', fontweight='bold')
        ax6.set_title('Overall Performance (Weighted)')
        ax6.set_xlim([0, 1.1])
        
        for i, (model, score) in enumerate(scores):
            ax6.text(score + 0.01, i, f'{score:.4f}', va='center')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison plot saved: {save_path}")
        plt.close()
    
    def plot_confusion_matrices(self, save_path='outputs/visualizations/confusion_matrices.png'):
        """Plot confusion matrices for all models"""
        
        if not self.results:
            return
        
        models = list(self.results.keys())
        n_models = len(models)
        
        cols = 3
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        fig.suptitle('Confusion Matrices - All Models', fontsize=16, fontweight='bold')
        
        axes = axes.flatten() if n_models > 1 else [axes]
        
        for i, model in enumerate(models):
            cm = self.results[model]['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                       cbar_kws={'label': 'Count'})
            axes[i].set_xlabel('Predicted', fontweight='bold')
            axes[i].set_ylabel('True', fontweight='bold')
            axes[i].set_title(f'{model}\nF1: {self.results[model]["f1_score"]:.4f}')
            axes[i].set_xticklabels(['Non-Outbreak', 'Outbreak'])
            axes[i].set_yticklabels(['Non-Outbreak', 'Outbreak'])
        
        # Hide extra subplots
        for i in range(n_models, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrices saved: {save_path}")
        plt.close()
    
    def generate_recommendation(self):
        """
        Generate recommendation for best model based on multiple criteria
        """
        if not self.results:
            return None
        
        models = list(self.results.keys())
        
        # Scoring criteria
        scores = {}
        for model in models:
            # Performance score (70% weight)
            performance = (
                0.4 * self.results[model]['f1_score'] +
                0.3 * self.results[model]['precision'] +
                0.2 * self.results[model]['recall'] +
                0.1 * self.results[model]['specificity']
            )
            
            # Speed score (15% weight) - normalized inverse of inference time
            if self.results[model]['inference_time_ms']:
                times = [self.results[m]['inference_time_ms'] 
                        for m in models if self.results[m]['inference_time_ms']]
                max_time = max(times)
                speed_score = 1 - (self.results[model]['inference_time_ms'] / max_time)
            else:
                speed_score = 0.5
            
            # Efficiency score (15% weight) - normalized inverse of model size
            if self.results[model]['model_size_mb']:
                sizes = [self.results[m]['model_size_mb'] 
                        for m in models if self.results[m]['model_size_mb']]
                max_size = max(sizes)
                efficiency_score = 1 - (self.results[model]['model_size_mb'] / max_size)
            else:
                efficiency_score = 0.5
            
            # Combined score
            total_score = (0.70 * performance + 
                          0.15 * speed_score + 
                          0.15 * efficiency_score)
            
            scores[model] = {
                'total_score': total_score,
                'performance_score': performance,
                'speed_score': speed_score,
                'efficiency_score': efficiency_score
            }
        
        # Best model
        best_model = max(scores, key=lambda m: scores[m]['total_score'])
        
        recommendation = {
            'recommended_model': best_model,
            'total_score': scores[best_model]['total_score'],
            'reasons': [],
            'all_scores': scores
        }
        
        # Generate reasons
        if self.results[best_model]['f1_score'] == max(self.results[m]['f1_score'] for m in models):
            recommendation['reasons'].append("Highest F1-Score")
        
        if self.results[best_model]['precision'] == max(self.results[m]['precision'] for m in models):
            recommendation['reasons'].append("Best Precision (fewer false alarms)")
        
        if self.results[best_model]['recall'] == max(self.results[m]['recall'] for m in models):
            recommendation['reasons'].append("Best Recall (detects more outbreaks)")
        
        return recommendation
    
    def save_results(self, filepath='outputs/model_comparison_results.json'):
        """Save all results to JSON file"""
        
        # Convert numpy types to Python types for JSON serialization
        results_serializable = {}
        for model, metrics in self.results.items():
            results_serializable[model] = {}
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    results_serializable[model][key] = value.tolist()
                elif isinstance(value, (np.int64, np.int32)):
                    results_serializable[model][key] = int(value)
                elif isinstance(value, (np.float64, np.float32)):
                    results_serializable[model][key] = float(value)
                else:
                    results_serializable[model][key] = value
        
        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=4)
        
        print(f"✓ Results saved: {filepath}")
    
    def print_summary(self):
        """Print comprehensive summary"""
        
        if not self.results:
            print("No results available!")
            return
        
        print("\n" + "="*80)
        print(" " * 25 + "MODEL EVALUATION SUMMARY")
        print("="*80 + "\n")
        
        # Comparison table
        df = self.get_comparison_table()
        print(df.to_string(index=False))
        
        print("\n" + "-"*80 + "\n")
        
        # Recommendation
        recommendation = self.generate_recommendation()
        if recommendation:
            print("🏆 RECOMMENDED MODEL FOR EPIWATCH APPLICATION")
            print("-"*80)
            print(f"Model: {recommendation['recommended_model']}")
            print(f"Overall Score: {recommendation['total_score']:.4f}")
            print(f"\nReasons:")
            for reason in recommendation['reasons']:
                print(f"  ✓ {reason}")
            
            best_model = recommendation['recommended_model']
            print(f"\nKey Metrics:")
            print(f"  • Accuracy: {self.results[best_model]['accuracy']:.4f}")
            print(f"  • Precision: {self.results[best_model]['precision']:.4f}")
            print(f"  • Recall: {self.results[best_model]['recall']:.4f}")
            print(f"  • F1-Score: {self.results[best_model]['f1_score']:.4f}")
            
            if self.results[best_model]['inference_time_ms']:
                print(f"  • Inference Time: {self.results[best_model]['inference_time_ms']:.2f} ms")
        
        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    # Example usage
    print("Model Evaluator for EpiWatch")
    
    # Dummy data for demonstration
    evaluator = ModelEvaluator()
    
    # Example: Evaluate a model
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
    y_pred = np.array([0, 1, 1, 0, 1, 0, 0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.8, 0.2, 0.95, 0.3, 0.45, 0.88, 0.15, 0.6])
    
    evaluator.evaluate_model(
        "Example Model",
        y_true, y_pred, y_prob,
        inference_time=0.05,
        model_size=400
    )
    
    print(evaluator.get_comparison_table())
