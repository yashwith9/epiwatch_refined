"""
Generate comprehensive metrics table for all NLP models
"""
import json

# Load results
with open('results/ultrafast_results.json', 'r') as f:
    results = json.load(f)

print("=" * 100)
print("📊 COMPREHENSIVE MODEL PERFORMANCE METRICS")
print("=" * 100)

# Main metrics table
print("\n🎯 ACCURACY & F1 SCORE")
print("-" * 100)
print(f"{'Model':<25} {'Accuracy':<15} {'F1 Score':<15} {'Status':<15}")
print("-" * 100)

for model_name, metrics in results.items():
    accuracy = f"{metrics['accuracy']:.2%}" if metrics['accuracy'] else "N/A"
    f1 = f"{metrics['f1']:.2%}" if metrics['f1'] else "N/A"
    status = "✅ " + metrics['status'].upper() if metrics['status'] == 'success' else "❌ FAILED"
    print(f"{model_name:<25} {accuracy:<15} {f1:<15} {status:<15}")

# Training performance
print("\n⏱️  TRAINING & EVALUATION TIME")
print("-" * 100)
print(f"{'Model':<25} {'Training Time':<20} {'Eval Time':<20} {'Total Time':<15}")
print("-" * 100)

for model_name, metrics in results.items():
    train_time = f"{metrics['train_time']:.2f}s"
    eval_time = f"{metrics['eval_time']:.2f}s"
    total_time = f"{metrics['train_time'] + metrics['eval_time']:.2f}s"
    print(f"{model_name:<25} {train_time:<20} {eval_time:<20} {total_time:<15}")

# Inference speed
print("\n⚡ INFERENCE SPEED")
print("-" * 100)
print(f"{'Model':<25} {'Inference Time (ms)':<25} {'Speed Rank':<15}")
print("-" * 100)

# Sort by inference time
sorted_models = sorted(results.items(), key=lambda x: x[1]['inference_ms'])
for rank, (model_name, metrics) in enumerate(sorted_models, 1):
    inference = f"{metrics['inference_ms']:.2f} ms"
    speed_indicator = "🚀 FASTEST" if rank == 1 else "⚡ FAST" if rank <= 2 else "🐢 SLOW" if rank > 4 else "→ AVERAGE"
    print(f"{model_name:<25} {inference:<25} {speed_indicator:<15}")

# Overall ranking
print("\n🏆 OVERALL MODEL RANKING")
print("-" * 100)
print(f"{'Rank':<8} {'Model':<25} {'Accuracy':<12} {'F1':<12} {'Speed':<15} {'Rating':<15}")
print("-" * 100)

# Calculate overall score (accuracy * 0.5 + f1 * 0.3 + speed_score * 0.2)
model_scores = []
for model_name, metrics in results.items():
    accuracy = metrics['accuracy'] if metrics['accuracy'] else 0
    f1 = metrics['f1'] if metrics['f1'] else 0
    # Speed score (inverse - faster is better)
    max_inference = max(m['inference_ms'] for m in results.values())
    speed_score = 1 - (metrics['inference_ms'] / max_inference)
    
    overall_score = (accuracy * 0.5) + (f1 * 0.3) + (speed_score * 0.2)
    model_scores.append((model_name, metrics, overall_score))

# Sort by overall score
model_scores.sort(key=lambda x: x[2], reverse=True)

for rank, (model_name, metrics, score) in enumerate(model_scores, 1):
    accuracy = f"{metrics['accuracy']:.2%}"
    f1 = f"{metrics['f1']:.2%}"
    speed = f"{metrics['inference_ms']:.1f}ms"
    rating = "⭐⭐⭐⭐⭐" if rank == 1 else "⭐⭐⭐⭐" if rank == 2 else "⭐⭐⭐" if rank == 3 else "⭐⭐"
    print(f"{rank:<8} {model_name:<25} {accuracy:<12} {f1:<12} {speed:<15} {rating:<15}")

# Summary statistics
print("\n📈 SUMMARY STATISTICS")
print("-" * 100)

accuracies = [m['accuracy'] for m in results.values() if m['accuracy']]
f1_scores = [m['f1'] for m in results.values() if m['f1']]
train_times = [m['train_time'] for m in results.values()]
inference_times = [m['inference_ms'] for m in results.values()]

print(f"Average Accuracy:        {sum(accuracies)/len(accuracies):.2%}")
print(f"Average F1 Score:        {sum(f1_scores)/len(f1_scores):.2%}")
print(f"Average Training Time:   {sum(train_times)/len(train_times):.2f}s")
print(f"Average Inference Time:  {sum(inference_times)/len(inference_times):.2f}ms")
print(f"\nBest Accuracy:           {max(accuracies):.2%} (DistilBERT, mBERT)")
print(f"Best F1 Score:           {max(f1_scores):.2%} (DistilBERT, mBERT)")
print(f"Fastest Training:        {min(train_times):.2f}s (Custom LSTM+Attention)")
print(f"Fastest Inference:       {min(inference_times):.2f}ms (Custom LSTM+Attention)")

# Model recommendations
print("\n💡 MODEL RECOMMENDATIONS")
print("-" * 100)
print("🥇 BEST OVERALL:         DistilBERT (100% accuracy, fast inference)")
print("🥈 BEST SPEED:           Custom LSTM+Attention (5ms inference, 82% accuracy)")
print("🥉 BEST BALANCED:        mBERT (100% accuracy, good multilingual support)")
print("⚠️  NEEDS IMPROVEMENT:   MuRIL, XLM-RoBERTa (low accuracy, need more training)")

print("\n" + "=" * 100)
print("✅ METRICS ANALYSIS COMPLETE")
print("=" * 100)
