"""
Complete Training Pipeline for All 5 Models
Train custom model + 4 pre-trained models and compare
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
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from preprocessing.text_preprocessing import TextPreprocessor, DatasetBuilder
from models.custom_model import CustomEpiDetector, ModelTrainer, build_vocab, EpidemicDataset
from models.pretrained_models import PretrainedEpiDetector, MODEL_CONFIGS
from evaluation.model_evaluator import ModelEvaluator
from evaluation.anomaly_detection import OutbreakAlertSystem
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


class EpiWatchTrainingPipeline:
    """
    Complete end-to-end training pipeline for EpiWatch
    """
    
    def __init__(self, data_path='data/processed/epidemic_data.csv',
                 output_dir='outputs'):
        """
        Args:
            data_path: Path to training data
            output_dir: Directory for outputs
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("="*80)
        print(" " * 25 + "EPIWATCH TRAINING PIPELINE")
        print("="*80)
        print(f"\n✓ Device: {self.device}")
        print(f"✓ Data path: {data_path}")
        print(f"✓ Output directory: {output_dir}\n")
        
        # Initialize components
        self.preprocessor = TextPreprocessor()
        self.dataset_builder = DatasetBuilder(self.preprocessor)
        self.evaluator = ModelEvaluator()
        
        # Create output directories
        os.makedirs(f"{output_dir}/visualizations", exist_ok=True)
        os.makedirs(f"{output_dir}/alerts", exist_ok=True)
        os.makedirs("models/saved", exist_ok=True)
    
    def load_and_prepare_data(self):
        """Load and prepare dataset"""
        print("\n" + "="*80)
        print("STEP 1: DATA LOADING AND PREPROCESSING")
        print("="*80 + "\n")
        
        # Check if data file exists
        if not os.path.exists(self.data_path):
            print(f"⚠️  Data file not found: {self.data_path}")
            print("Creating sample dataset for demonstration...\n")
            
            # Create sample dataset
            df = self.dataset_builder.create_sample_dataset(
                n_samples=2000,
                save_path=self.data_path
            )
        else:
            print(f"Loading data from {self.data_path}...")
            df = self.dataset_builder.load_data(self.data_path)
            print(f"✓ Loaded {len(df)} samples\n")
        
        # Display data info
        print(f"Dataset shape: {df.shape}")
        print(f"\nColumns: {df.columns.tolist()}")
        print(f"\nClass distribution:\n{df['label'].value_counts()}\n")
        
        # Preprocess texts
        print("Preprocessing texts...")
        df['processed_text'] = self.preprocessor.preprocess_dataset(
            df['text'].tolist(), 
            show_progress=True
        )
        
        # Balance dataset
        df = self.dataset_builder.balance_dataset(df, text_col='processed_text')
        
        # Split dataset
        self.data_splits = self.dataset_builder.prepare_train_test_split(
            df, 
            text_col='processed_text',
            test_size=0.2,
            val_size=0.1
        )
        
        print("\n✓ Data preparation complete!\n")
        return df
    
    def train_custom_model(self):
        """Train custom neural network from scratch"""
        print("\n" + "="*80)
        print("STEP 2: TRAINING CUSTOM NEURAL NETWORK (FROM SCRATCH)")
        print("="*80 + "\n")
        
        # Build vocabulary
        print("Building vocabulary...")
        vocab = build_vocab(self.data_splits['train']['texts'], min_freq=2)
        print(f"✓ Vocabulary size: {len(vocab)}\n")
        
        # Create datasets
        train_dataset = EpidemicDataset(
            self.data_splits['train']['texts'],
            self.data_splits['train']['labels'],
            vocab,
            max_length=256
        )
        
        val_dataset = EpidemicDataset(
            self.data_splits['val']['texts'],
            self.data_splits['val']['labels'],
            vocab,
            max_length=256
        )
        
        test_dataset = EpidemicDataset(
            self.data_splits['test']['texts'],
            self.data_splits['test']['labels'],
            vocab,
            max_length=256
        )
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Initialize model
        model = CustomEpiDetector(
            vocab_size=len(vocab),
            embedding_dim=256,
            hidden_dim=128,
            num_layers=2,
            dropout=0.3
        )
        
        print(f"Model architecture:\n{model}\n")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")
        
        # Train
        trainer = ModelTrainer(model, device=self.device)
        trainer.train(train_loader, val_loader, epochs=10, lr=0.001)
        
        # Evaluate
        print("\nEvaluating on test set...")
        predictions, _ = trainer.predict(test_loader)
        y_pred = (predictions > 0.5).astype(int)
        y_true = np.array(self.data_splits['test']['labels'])
        
        self.evaluator.evaluate_model(
            "Custom Neural Network",
            y_true,
            y_pred,
            y_prob=predictions,
            inference_time=0.05,  # Approximate
            model_size=50  # Approximate MB
        )
        
        print("\n✓ Custom model training complete!\n")
        
        # Save vocab
        with open('models/saved/vocab.json', 'w') as f:
            json.dump(vocab, f)
    
    def train_pretrained_model(self, model_key):
        """Train a single pre-trained model"""
        config = MODEL_CONFIGS[model_key]
        model_name = config['name']
        
        print(f"\n{'='*80}")
        print(f"Training: {model_name}")
        print(f"Description: {config['description']}")
        print(f"{'='*80}\n")
        
        # Initialize model
        model = PretrainedEpiDetector(model_name, device=self.device)
        
        # Prepare dataloaders
        train_loader = model.prepare_dataloader(
            self.data_splits['train']['texts'],
            self.data_splits['train']['labels'],
            batch_size=16,
            max_length=512
        )
        
        val_loader = model.prepare_dataloader(
            self.data_splits['val']['texts'],
            self.data_splits['val']['labels'],
            batch_size=16,
            max_length=512,
            shuffle=False
        )
        
        # Train
        model.train(train_loader, val_loader, epochs=3, learning_rate=2e-5)
        
        # Evaluate
        print("\nEvaluating on test set...")
        predictions, probabilities = model.predict(
            self.data_splits['test']['texts'],
            batch_size=16
        )
        
        y_true = np.array(self.data_splits['test']['labels'])
        
        # Measure inference time
        sample_text = self.data_splits['test']['texts'][0]
        timing = model.measure_inference_time(sample_text, num_runs=50)
        
        # Model size (approximate)
        model_sizes = {
            'xlm-roberta-base': 550,
            'bert-base-multilingual-cased': 680,
            'distilbert-base-multilingual-cased': 270,
            'google/muril-base-cased': 890
        }
        
        self.evaluator.evaluate_model(
            model_name,
            y_true,
            predictions,
            y_prob=probabilities,
            inference_time=timing['mean'],
            model_size=model_sizes.get(model_name, 500)
        )
        
        print(f"\n✓ {model_name} training complete!\n")
    
    def train_all_pretrained_models(self):
        """Train all 4 pre-trained models"""
        print("\n" + "="*80)
        print("STEP 3: TRAINING PRE-TRAINED TRANSFORMER MODELS")
        print("="*80 + "\n")
        
        for model_key in MODEL_CONFIGS.keys():
            try:
                self.train_pretrained_model(model_key)
            except Exception as e:
                print(f"\n❌ Error training {model_key}: {str(e)}\n")
                continue
    
    def compare_and_visualize(self):
        """Compare all models and create visualizations"""
        print("\n" + "="*80)
        print("STEP 4: MODEL COMPARISON AND VISUALIZATION")
        print("="*80 + "\n")
        
        # Print summary
        self.evaluator.print_summary()
        
        # Generate comparison table
        comparison_df = self.evaluator.get_comparison_table()
        comparison_df.to_csv(f"{self.output_dir}/model_comparison_table.csv", index=False)
        print(f"\n✓ Comparison table saved: {self.output_dir}/model_comparison_table.csv")
        
        # Create visualizations
        self.evaluator.plot_comparison(
            save_path=f"{self.output_dir}/visualizations/model_comparison.png"
        )
        
        self.evaluator.plot_confusion_matrices(
            save_path=f"{self.output_dir}/visualizations/confusion_matrices.png"
        )
        
        # Save results
        self.evaluator.save_results(
            filepath=f"{self.output_dir}/model_comparison_results.json"
        )
        
        # Get recommendation
        recommendation = self.evaluator.generate_recommendation()
        
        with open(f"{self.output_dir}/recommendation.json", 'w') as f:
            json.dump(recommendation, f, indent=4)
        
        print(f"\n✓ All results saved to {self.output_dir}/\n")
        
        return recommendation
    
    def run_complete_pipeline(self):
        """Run the complete training and evaluation pipeline"""
        start_time = datetime.now()
        
        print("\n" + "🚀 " * 40)
        print("STARTING COMPLETE EPIWATCH TRAINING PIPELINE")
        print("🚀 " * 40 + "\n")
        
        try:
            # Step 1: Load and prepare data
            df = self.load_and_prepare_data()
            
            # Step 2: Train custom model
            self.train_custom_model()
            
            # Step 3: Train pre-trained models
            self.train_all_pretrained_models()
            
            # Step 4: Compare and visualize
            recommendation = self.compare_and_visualize()
            
            # Success message
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds() / 60
            
            print("\n" + "🎉 " * 40)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("🎉 " * 40 + "\n")
            
            print(f"⏱️  Total time: {duration:.2f} minutes")
            print(f"🏆 Recommended model: {recommendation['recommended_model']}")
            print(f"📊 Check outputs in: {self.output_dir}/")
            print(f"📈 Visualizations: {self.output_dir}/visualizations/")
            
            print("\n" + "="*80 + "\n")
            
            return recommendation
            
        except Exception as e:
            print(f"\n❌ Pipeline failed with error: {str(e)}\n")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Main entry point"""
    # Initialize pipeline
    pipeline = EpiWatchTrainingPipeline(
        data_path='data/processed/epidemic_data.csv',
        output_dir='outputs'
    )
    
    # Run complete pipeline
    recommendation = pipeline.run_complete_pipeline()
    
    if recommendation:
        print(f"\n✅ Training complete! Use {recommendation['recommended_model']} for your mobile app.\n")
    else:
        print("\n❌ Training failed. Please check errors above.\n")


if __name__ == "__main__":
    main()
