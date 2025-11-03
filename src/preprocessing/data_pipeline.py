"""
Core Data Processing Pipeline for Model Comparison System
Handles dataset loading, preprocessing, and splitting for epidemic detection models
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import json
import logging
from sklearn.model_selection import train_test_split, StratifiedKFold
from dataclasses import dataclass
from datetime import datetime

from .text_preprocessing import TextPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EpidemicSample:
    """Data structure for epidemic detection samples"""
    id: str
    text: str
    label: int  # 0: no epidemic signal, 1: epidemic signal
    disease_type: Optional[str] = None
    location: Optional[str] = None
    timestamp: Optional[datetime] = None
    confidence: Optional[float] = None


class DatasetLoader:
    """
    Handles loading and validation of datasets from various file formats
    Supports CSV, JSON, and text files for epidemic detection data
    """
    
    def __init__(self):
        self.supported_formats = ['.csv', '.json', '.txt', '.tsv']
        logger.info("DatasetLoader initialized")
    
    def load_dataset(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load dataset from file with automatic format detection
        
        Args:
            file_path: Path to the dataset file
            **kwargs: Additional arguments for pandas readers
            
        Returns:
            DataFrame containing the loaded dataset
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        file_ext = file_path.suffix.lower()
        
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_ext}. Supported: {self.supported_formats}")
        
        logger.info(f"Loading dataset from: {file_path}")
        
        try:
            if file_ext == '.csv':
                df = self._load_csv(file_path, **kwargs)
            elif file_ext == '.json':
                df = self._load_json(file_path, **kwargs)
            elif file_ext == '.txt':
                df = self._load_text(file_path, **kwargs)
            elif file_ext == '.tsv':
                df = self._load_tsv(file_path, **kwargs)
            
            logger.info(f"Successfully loaded {len(df)} samples from {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def _load_csv(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load CSV file"""
        default_kwargs = {'encoding': 'utf-8'}
        default_kwargs.update(kwargs)
        return pd.read_csv(file_path, **default_kwargs)
    
    def _load_json(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load JSON file"""
        default_kwargs = {'encoding': 'utf-8'}
        default_kwargs.update(kwargs)
        
        with open(file_path, 'r', **default_kwargs) as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            if 'data' in data:
                return pd.DataFrame(data['data'])
            else:
                return pd.DataFrame([data])
        else:
            raise ValueError("Unsupported JSON structure")
    
    def _load_text(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load plain text file (one sample per line)"""
        default_kwargs = {'encoding': 'utf-8'}
        default_kwargs.update(kwargs)
        
        with open(file_path, 'r', **default_kwargs) as f:
            lines = f.readlines()
        
        # Create DataFrame with text and default labels
        texts = [line.strip() for line in lines if line.strip()]
        return pd.DataFrame({
            'text': texts,
            'label': [0] * len(texts)  # Default to no epidemic signal
        })
    
    def _load_tsv(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load TSV file"""
        default_kwargs = {'sep': '\t', 'encoding': 'utf-8'}
        default_kwargs.update(kwargs)
        return pd.read_csv(file_path, **default_kwargs)
    
    def validate_dataset(self, df: pd.DataFrame, 
                        text_col: str = 'text', 
                        label_col: str = 'label') -> Dict[str, any]:
        """
        Validate dataset structure and content
        
        Args:
            df: DataFrame to validate
            text_col: Name of text column
            label_col: Name of label column
            
        Returns:
            Dictionary with validation results and statistics
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check required columns
        required_cols = [text_col, label_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_cols}")
        
        if not validation_results['is_valid']:
            return validation_results
        
        # Check for empty values
        empty_text = df[text_col].isna().sum() + (df[text_col] == '').sum()
        empty_labels = df[label_col].isna().sum()
        
        if empty_text > 0:
            validation_results['warnings'].append(f"Found {empty_text} empty text entries")
        
        if empty_labels > 0:
            validation_results['errors'].append(f"Found {empty_labels} missing labels")
            validation_results['is_valid'] = False
        
        # Check label values
        unique_labels = df[label_col].unique()
        if not all(label in [0, 1] for label in unique_labels if not pd.isna(label)):
            validation_results['warnings'].append(f"Non-binary labels found: {unique_labels}")
        
        # Calculate statistics
        validation_results['statistics'] = {
            'total_samples': len(df),
            'unique_texts': df[text_col].nunique(),
            'label_distribution': df[label_col].value_counts().to_dict(),
            'avg_text_length': df[text_col].str.len().mean(),
            'empty_texts': empty_text,
            'duplicate_texts': df[text_col].duplicated().sum()
        }
        
        logger.info(f"Dataset validation completed. Valid: {validation_results['is_valid']}")
        return validation_results
    
    def load_multiple_datasets(self, file_paths: List[str], 
                             combine: bool = True) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """
        Load multiple datasets
        
        Args:
            file_paths: List of file paths to load
            combine: Whether to combine all datasets into one DataFrame
            
        Returns:
            Combined DataFrame or list of DataFrames
        """
        datasets = []
        
        for file_path in file_paths:
            try:
                df = self.load_dataset(file_path)
                datasets.append(df)
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                continue
        
        if combine and datasets:
            combined_df = pd.concat(datasets, ignore_index=True)
            logger.info(f"Combined {len(datasets)} datasets into {len(combined_df)} samples")
            return combined_df
        
        return datasets


class DataSplitter:
    """
    Utilities for splitting datasets into train/validation/test sets
    Supports various splitting strategies for epidemic detection data
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        logger.info(f"DataSplitter initialized with random_state={random_state}")
    
    def create_splits(self, 
                     df: pd.DataFrame,
                     test_size: float = 0.2,
                     val_size: float = 0.1,
                     text_col: str = 'text',
                     label_col: str = 'label',
                     stratify: bool = True) -> Dict[str, Dict[str, List]]:
        """
        Split dataset into train, validation, and test sets
        
        Args:
            df: Input DataFrame
            test_size: Proportion of test set (0.0 to 1.0)
            val_size: Proportion of validation set (0.0 to 1.0)
            text_col: Name of text column
            label_col: Name of label column
            stratify: Whether to maintain label distribution across splits
            
        Returns:
            Dictionary with train, val, test splits containing texts and labels
        """
        if test_size + val_size >= 1.0:
            raise ValueError("test_size + val_size must be less than 1.0")
        
        # Prepare stratification
        stratify_col = df[label_col] if stratify else None
        
        # First split: separate test set
        if test_size > 0:
            train_val_df, test_df = train_test_split(
                df, 
                test_size=test_size, 
                random_state=self.random_state,
                stratify=stratify_col
            )
        else:
            train_val_df = df.copy()
            test_df = pd.DataFrame()
        
        # Second split: separate validation set from remaining data
        if val_size > 0 and len(train_val_df) > 0:
            # Adjust validation size relative to remaining data
            adjusted_val_size = val_size / (1 - test_size)
            stratify_col_remaining = train_val_df[label_col] if stratify else None
            
            train_df, val_df = train_test_split(
                train_val_df,
                test_size=adjusted_val_size,
                random_state=self.random_state,
                stratify=stratify_col_remaining
            )
        else:
            train_df = train_val_df.copy()
            val_df = pd.DataFrame()
        
        # Create splits dictionary
        splits = {
            'train': {
                'texts': train_df[text_col].tolist() if len(train_df) > 0 else [],
                'labels': train_df[label_col].tolist() if len(train_df) > 0 else []
            },
            'val': {
                'texts': val_df[text_col].tolist() if len(val_df) > 0 else [],
                'labels': val_df[label_col].tolist() if len(val_df) > 0 else []
            },
            'test': {
                'texts': test_df[text_col].tolist() if len(test_df) > 0 else [],
                'labels': test_df[label_col].tolist() if len(test_df) > 0 else []
            }
        }
        
        # Log split information
        logger.info("Dataset splits created:")
        logger.info(f"  Train: {len(splits['train']['texts'])} samples")
        logger.info(f"  Validation: {len(splits['val']['texts'])} samples")
        logger.info(f"  Test: {len(splits['test']['texts'])} samples")
        
        # Log label distribution if stratified
        if stratify:
            for split_name, split_data in splits.items():
                if split_data['labels']:
                    labels_array = np.array(split_data['labels'])
                    unique, counts = np.unique(labels_array, return_counts=True)
                    distribution = dict(zip(unique, counts))
                    logger.info(f"  {split_name.capitalize()} label distribution: {distribution}")
        
        return splits
    
    def create_cross_validation_splits(self, 
                                     df: pd.DataFrame,
                                     n_splits: int = 5,
                                     text_col: str = 'text',
                                     label_col: str = 'label',
                                     stratify: bool = True) -> List[Dict[str, Dict[str, List]]]:
        """
        Create cross-validation splits
        
        Args:
            df: Input DataFrame
            n_splits: Number of CV folds
            text_col: Name of text column
            label_col: Name of label column
            stratify: Whether to use stratified CV
            
        Returns:
            List of dictionaries, each containing train and val splits for one fold
        """
        if stratify:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            split_generator = cv.split(df[text_col], df[label_col])
        else:
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            split_generator = cv.split(df[text_col])
        
        cv_splits = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(split_generator):
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]
            
            fold_split = {
                'train': {
                    'texts': train_df[text_col].tolist(),
                    'labels': train_df[label_col].tolist()
                },
                'val': {
                    'texts': val_df[text_col].tolist(),
                    'labels': val_df[label_col].tolist()
                }
            }
            
            cv_splits.append(fold_split)
            logger.info(f"Fold {fold_idx + 1}: Train={len(fold_split['train']['texts'])}, Val={len(fold_split['val']['texts'])}")
        
        return cv_splits
    
    def temporal_split(self, 
                      df: pd.DataFrame,
                      date_col: str,
                      test_months: int = 2,
                      val_months: int = 1,
                      text_col: str = 'text',
                      label_col: str = 'label') -> Dict[str, Dict[str, List]]:
        """
        Create temporal splits based on date column
        
        Args:
            df: Input DataFrame with date column
            date_col: Name of date column
            test_months: Number of most recent months for test set
            val_months: Number of months before test set for validation
            text_col: Name of text column
            label_col: Name of label column
            
        Returns:
            Dictionary with train, val, test splits
        """
        # Convert date column to datetime
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Sort by date
        df = df.sort_values(date_col)
        
        # Calculate split dates
        max_date = df[date_col].max()
        test_start_date = max_date - pd.DateOffset(months=test_months)
        val_start_date = test_start_date - pd.DateOffset(months=val_months)
        
        # Create splits
        test_df = df[df[date_col] >= test_start_date]
        val_df = df[(df[date_col] >= val_start_date) & (df[date_col] < test_start_date)]
        train_df = df[df[date_col] < val_start_date]
        
        splits = {
            'train': {
                'texts': train_df[text_col].tolist(),
                'labels': train_df[label_col].tolist()
            },
            'val': {
                'texts': val_df[text_col].tolist(),
                'labels': val_df[label_col].tolist()
            },
            'test': {
                'texts': test_df[text_col].tolist(),
                'labels': test_df[label_col].tolist()
            }
        }
        
        logger.info("Temporal splits created:")
        logger.info(f"  Train: {len(splits['train']['texts'])} samples (before {val_start_date.date()})")
        logger.info(f"  Validation: {len(splits['val']['texts'])} samples ({val_start_date.date()} to {test_start_date.date()})")
        logger.info(f"  Test: {len(splits['test']['texts'])} samples (after {test_start_date.date()})")
        
        return splits


class DataPipeline:
    """
    Complete data processing pipeline combining loading, preprocessing, and splitting
    """
    
    def __init__(self, preprocessor: Optional[TextPreprocessor] = None):
        self.loader = DatasetLoader()
        self.splitter = DataSplitter()
        self.preprocessor = preprocessor or TextPreprocessor()
        logger.info("DataPipeline initialized")
    
    def process_dataset(self, 
                       file_path: str,
                       text_col: str = 'text',
                       label_col: str = 'label',
                       test_size: float = 0.2,
                       val_size: float = 0.1,
                       preprocess_text: bool = True,
                       validate: bool = True) -> Dict[str, any]:
        """
        Complete pipeline: load, validate, preprocess, and split dataset
        
        Args:
            file_path: Path to dataset file
            text_col: Name of text column
            label_col: Name of label column
            test_size: Proportion of test set
            val_size: Proportion of validation set
            preprocess_text: Whether to preprocess text data
            validate: Whether to validate dataset
            
        Returns:
            Dictionary containing splits and metadata
        """
        # Load dataset
        df = self.loader.load_dataset(file_path)
        
        # Validate dataset
        if validate:
            validation_results = self.loader.validate_dataset(df, text_col, label_col)
            if not validation_results['is_valid']:
                raise ValueError(f"Dataset validation failed: {validation_results['errors']}")
        
        # Preprocess text if requested
        if preprocess_text:
            logger.info("Preprocessing text data...")
            df[text_col] = df[text_col].apply(self.preprocessor.preprocess)
        
        # Create splits
        splits = self.splitter.create_splits(
            df, test_size=test_size, val_size=val_size, 
            text_col=text_col, label_col=label_col
        )
        
        # Prepare result
        result = {
            'splits': splits,
            'metadata': {
                'source_file': file_path,
                'total_samples': len(df),
                'text_column': text_col,
                'label_column': label_col,
                'preprocessed': preprocess_text,
                'validation_results': validation_results if validate else None
            }
        }
        
        logger.info("Data pipeline processing completed successfully")
        return result
    
    def save_splits(self, splits: Dict[str, Dict[str, List]], output_dir: str):
        """
        Save dataset splits to files
        
        Args:
            splits: Dictionary containing train/val/test splits
            output_dir: Directory to save split files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for split_name, split_data in splits.items():
            if split_data['texts']:  # Only save non-empty splits
                split_df = pd.DataFrame({
                    'text': split_data['texts'],
                    'label': split_data['labels']
                })
                
                file_path = output_path / f"{split_name}.csv"
                split_df.to_csv(file_path, index=False)
                logger.info(f"Saved {split_name} split to {file_path}")


if __name__ == "__main__":
    # Example usage
    print("Core Data Processing Pipeline")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = DataPipeline()
    
    # Create sample dataset for testing
    from .text_preprocessing import DatasetBuilder
    builder = DatasetBuilder()
    sample_df = builder.create_sample_dataset(n_samples=1000)
    sample_path = "data/sample_epidemic_data.csv"
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    sample_df.to_csv(sample_path, index=False)
    
    # Process dataset
    try:
        result = pipeline.process_dataset(
            file_path=sample_path,
            test_size=0.2,
            val_size=0.1,
            preprocess_text=True
        )
        
        print("\nProcessing completed successfully!")
        print(f"Total samples: {result['metadata']['total_samples']}")
        print(f"Train samples: {len(result['splits']['train']['texts'])}")
        print(f"Validation samples: {len(result['splits']['val']['texts'])}")
        print(f"Test samples: {len(result['splits']['test']['texts'])}")
        
    except Exception as e:
        print(f"Error: {e}")