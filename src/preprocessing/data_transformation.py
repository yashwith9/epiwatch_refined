"""
Dataset Transformation Utilities for Model Comparison System
Converts structured outbreak data to text format for NLP training
Creates synthetic text generation and data augmentation techniques
"""

import pandas as pd
import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import logging
from datetime import datetime, timedelta
import json
import re
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OutbreakRecord:
    """Structured outbreak data record"""
    id_outbreak: str
    year: int
    disease: str
    country: str
    iso3: str
    unsd_region: str
    who_region: str


class StructuredToTextConverter:
    """
    Converts structured outbreak data to natural language text for NLP training
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        random.seed(random_state)
        np.random.seed(random_state)
        
        # Text templates for different outbreak scenarios
        self.outbreak_templates = [
            "Outbreak of {disease} reported in {country} ({region})",
            "{disease} cases confirmed in {country}, {region} region",
            "Health authorities report {disease} outbreak in {country}",
            "Multiple cases of {disease} detected in {country}",
            "{disease} epidemic declared in {country}, {region}",
            "Disease surveillance confirms {disease} in {country}",
            "Public health alert: {disease} outbreak in {country}",
            "WHO reports {disease} cases in {country}, {region}",
            "{country} health ministry confirms {disease} outbreak",
            "Emergency response activated for {disease} in {country}"
        ]
        
        self.severity_modifiers = [
            "severe", "critical", "widespread", "rapidly spreading",
            "contained", "localized", "emerging", "ongoing"
        ]
        
        self.context_phrases = [
            "health officials are investigating",
            "vaccination campaign initiated",
            "quarantine measures implemented",
            "international support requested",
            "surveillance systems activated",
            "contact tracing underway",
            "medical supplies deployed",
            "emergency protocols activated"
        ]
        
        # Disease-specific information
        self.disease_info = {
            "COVID-19": {
                "symptoms": ["fever", "cough", "breathing difficulties", "fatigue"],
                "transmission": "respiratory droplets",
                "severity": ["mild", "moderate", "severe", "critical"]
            },
            "Dengue fever": {
                "symptoms": ["high fever", "headache", "muscle pain", "rash"],
                "transmission": "mosquito-borne",
                "severity": ["mild", "severe", "hemorrhagic"]
            },
            "Cholera": {
                "symptoms": ["diarrhea", "vomiting", "dehydration"],
                "transmission": "contaminated water",
                "severity": ["mild", "severe", "life-threatening"]
            },
            "Measles": {
                "symptoms": ["fever", "rash", "cough", "runny nose"],
                "transmission": "airborne",
                "severity": ["mild", "severe", "complicated"]
            },
            "Malaria": {
                "symptoms": ["fever", "chills", "headache", "nausea"],
                "transmission": "mosquito-borne",
                "severity": ["uncomplicated", "severe", "cerebral"]
            }
        }
        
        logger.info("StructuredToTextConverter initialized")
    
    def convert_record_to_text(self, record: OutbreakRecord, 
                              include_details: bool = True,
                              add_context: bool = True) -> str:
        """
        Convert a single outbreak record to natural language text
        
        Args:
            record: OutbreakRecord instance
            include_details: Whether to include disease-specific details
            add_context: Whether to add contextual information
            
        Returns:
            Natural language text describing the outbreak
        """
        # Select base template
        template = random.choice(self.outbreak_templates)
        
        # Clean disease name
        disease = self._clean_disease_name(record.disease)
        
        # Format base text
        text = template.format(
            disease=disease,
            country=record.country,
            region=record.unsd_region
        )
        
        # Add severity modifier
        if random.random() < 0.4:  # 40% chance
            severity = random.choice(self.severity_modifiers)
            text = f"{severity.capitalize()} {text.lower()}"
        
        # Add disease-specific details
        if include_details and disease in self.disease_info:
            if random.random() < 0.3:  # 30% chance
                symptoms = random.sample(
                    self.disease_info[disease]["symptoms"], 
                    k=min(2, len(self.disease_info[disease]["symptoms"]))
                )
                text += f". Symptoms include {', '.join(symptoms)}"
        
        # Add contextual information
        if add_context and random.random() < 0.5:  # 50% chance
            context = random.choice(self.context_phrases)
            text += f". {context.capitalize()}"
        
        # Add case numbers (synthetic)
        if random.random() < 0.3:  # 30% chance
            case_count = random.randint(5, 500)
            text += f". {case_count} cases reported"
        
        return text
    
    def _clean_disease_name(self, disease: str) -> str:
        """Clean and normalize disease names"""
        # Remove codes and extra information
        disease = re.sub(r'[A-Z]\d+', '', disease)  # Remove ICD codes
        disease = re.sub(r',.*', '', disease)  # Remove everything after comma
        disease = disease.strip()
        
        # Handle specific cases
        if "COVID" in disease.upper():
            return "COVID-19"
        elif "dengue" in disease.lower():
            return "Dengue fever"
        elif "influenza" in disease.lower():
            return "Influenza"
        elif "yellow fever" in disease.lower():
            return "Yellow fever"
        elif "chikungunya" in disease.lower():
            return "Chikungunya"
        
        return disease
    
    def convert_dataframe_to_text(self, df: pd.DataFrame,
                                 text_col: str = 'text',
                                 label_col: str = 'label') -> pd.DataFrame:
        """
        Convert entire DataFrame of outbreak records to text format
        
        Args:
            df: DataFrame with outbreak records
            text_col: Name for text column
            label_col: Name for label column
            
        Returns:
            DataFrame with text and labels
        """
        texts = []
        labels = []
        
        for _, row in df.iterrows():
            record = OutbreakRecord(
                id_outbreak=row.get('id_outbreak', ''),
                year=row.get('Year', 2024),
                disease=row.get('Disease', ''),
                country=row.get('Country', ''),
                iso3=row.get('iso3', ''),
                unsd_region=row.get('unsd_region', ''),
                who_region=row.get('who_region', '')
            )
            
            text = self.convert_record_to_text(record)
            texts.append(text)
            labels.append(1)  # All outbreak records are positive examples
        
        result_df = pd.DataFrame({
            text_col: texts,
            label_col: labels
        })
        
        logger.info(f"Converted {len(result_df)} outbreak records to text")
        return result_df


class SyntheticTextGenerator:
    """
    Generates synthetic text data for epidemic detection training
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        random.seed(random_state)
        np.random.seed(random_state)
        
        # Positive (outbreak) text patterns
        self.positive_patterns = [
            "Breaking: {disease} outbreak in {location}",
            "Health alert: Multiple {disease} cases in {location}",
            "{disease} epidemic spreads across {location}",
            "Urgent: {disease} outbreak confirmed in {location}",
            "WHO declares {disease} emergency in {location}",
            "{location} reports surge in {disease} cases",
            "Public health crisis: {disease} in {location}",
            "Disease outbreak: {disease} affects {location}",
            "Emergency response: {disease} outbreak in {location}",
            "Health authorities confirm {disease} in {location}"
        ]
        
        # Negative (non-outbreak) text patterns
        self.negative_patterns = [
            "Health ministry announces vaccination program in {location}",
            "Medical conference held in {location}",
            "New hospital opens in {location}",
            "Health workers receive training in {location}",
            "Nutrition program launched in {location}",
            "Medical research facility established in {location}",
            "Health insurance coverage expanded in {location}",
            "Preventive healthcare campaign in {location}",
            "Medical equipment donated to {location}",
            "Health awareness week celebrated in {location}"
        ]
        
        # Disease names
        self.diseases = [
            "COVID-19", "Dengue fever", "Malaria", "Cholera", "Measles",
            "Influenza", "Yellow fever", "Chikungunya", "Zika virus",
            "Ebola", "MERS", "Tuberculosis", "Hepatitis", "Typhoid"
        ]
        
        # Location names
        self.locations = [
            "Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata",
            "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Lucknow",
            "Lagos", "Cairo", "Nairobi", "Accra", "Dakar",
            "Bangkok", "Manila", "Jakarta", "Kuala Lumpur", "Singapore",
            "São Paulo", "Mexico City", "Lima", "Bogotá", "Buenos Aires"
        ]
        
        # Additional context elements
        self.numbers = list(range(5, 1000))
        self.time_phrases = [
            "in the past week", "over the weekend", "since Monday",
            "in recent days", "this month", "yesterday", "today"
        ]
        
        logger.info("SyntheticTextGenerator initialized")
    
    def generate_positive_samples(self, n_samples: int) -> List[str]:
        """
        Generate positive (outbreak) text samples
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            List of positive text samples
        """
        samples = []
        
        for _ in range(n_samples):
            pattern = random.choice(self.positive_patterns)
            disease = random.choice(self.diseases)
            location = random.choice(self.locations)
            
            text = pattern.format(disease=disease, location=location)
            
            # Add additional details randomly
            if random.random() < 0.4:  # 40% chance
                case_count = random.choice(self.numbers)
                text += f". {case_count} cases reported"
            
            if random.random() < 0.3:  # 30% chance
                time_phrase = random.choice(self.time_phrases)
                text += f" {time_phrase}"
            
            samples.append(text)
        
        return samples
    
    def generate_negative_samples(self, n_samples: int) -> List[str]:
        """
        Generate negative (non-outbreak) text samples
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            List of negative text samples
        """
        samples = []
        
        for _ in range(n_samples):
            pattern = random.choice(self.negative_patterns)
            location = random.choice(self.locations)
            
            text = pattern.format(location=location)
            
            # Add additional context randomly
            if random.random() < 0.3:  # 30% chance
                time_phrase = random.choice(self.time_phrases)
                text += f" {time_phrase}"
            
            samples.append(text)
        
        return samples
    
    def generate_balanced_dataset(self, n_samples: int,
                                 positive_ratio: float = 0.5) -> pd.DataFrame:
        """
        Generate balanced synthetic dataset
        
        Args:
            n_samples: Total number of samples
            positive_ratio: Ratio of positive samples (0.0 to 1.0)
            
        Returns:
            DataFrame with synthetic text data
        """
        n_positive = int(n_samples * positive_ratio)
        n_negative = n_samples - n_positive
        
        positive_texts = self.generate_positive_samples(n_positive)
        negative_texts = self.generate_negative_samples(n_negative)
        
        # Combine and shuffle
        all_texts = positive_texts + negative_texts
        all_labels = [1] * n_positive + [0] * n_negative
        
        # Create DataFrame and shuffle
        df = pd.DataFrame({
            'text': all_texts,
            'label': all_labels
        })
        
        df = df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        logger.info(f"Generated {len(df)} synthetic samples ({n_positive} positive, {n_negative} negative)")
        return df


class DataAugmentor:
    """
    Data augmentation techniques for balanced datasets
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        random.seed(random_state)
        np.random.seed(random_state)
        
        # Synonym mappings for key terms
        self.synonyms = {
            "outbreak": ["epidemic", "surge", "increase", "spread", "emergence"],
            "cases": ["patients", "infections", "incidents", "occurrences"],
            "reported": ["confirmed", "detected", "identified", "documented"],
            "health": ["medical", "healthcare", "public health"],
            "authorities": ["officials", "ministry", "department", "agencies"],
            "emergency": ["crisis", "alert", "urgent situation", "critical situation"]
        }
        
        # Paraphrasing patterns
        self.paraphrase_patterns = [
            (r"(\d+) cases reported", r"reported \1 cases"),
            (r"outbreak in (\w+)", r"\1 experiences outbreak"),
            (r"health authorities", r"medical officials"),
            (r"confirmed cases", r"verified infections"),
            (r"disease outbreak", r"epidemic situation")
        ]
        
        logger.info("DataAugmentor initialized")
    
    def synonym_replacement(self, text: str, replacement_prob: float = 0.3) -> str:
        """
        Replace words with synonyms
        
        Args:
            text: Input text
            replacement_prob: Probability of replacing each word
            
        Returns:
            Text with synonym replacements
        """
        words = text.split()
        new_words = []
        
        for word in words:
            word_lower = word.lower().strip('.,!?')
            
            if (word_lower in self.synonyms and 
                random.random() < replacement_prob):
                synonym = random.choice(self.synonyms[word_lower])
                # Preserve capitalization
                if word[0].isupper():
                    synonym = synonym.capitalize()
                new_words.append(synonym)
            else:
                new_words.append(word)
        
        return ' '.join(new_words)
    
    def paraphrase_text(self, text: str) -> str:
        """
        Paraphrase text using pattern substitution
        
        Args:
            text: Input text
            
        Returns:
            Paraphrased text
        """
        result = text
        
        for pattern, replacement in self.paraphrase_patterns:
            if random.random() < 0.4:  # 40% chance to apply each pattern
                result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        return result
    
    def add_noise(self, text: str, noise_prob: float = 0.1) -> str:
        """
        Add minor noise to text (typos, extra spaces)
        
        Args:
            text: Input text
            noise_prob: Probability of adding noise
            
        Returns:
            Text with added noise
        """
        if random.random() > noise_prob:
            return text
        
        # Add extra spaces randomly
        if random.random() < 0.5:
            words = text.split()
            if len(words) > 1:
                idx = random.randint(0, len(words) - 1)
                words[idx] = words[idx] + " "
                text = ' '.join(words)
        
        return text
    
    def augment_text(self, text: str, 
                    use_synonyms: bool = True,
                    use_paraphrase: bool = True,
                    use_noise: bool = False) -> str:
        """
        Apply multiple augmentation techniques
        
        Args:
            text: Input text
            use_synonyms: Whether to use synonym replacement
            use_paraphrase: Whether to use paraphrasing
            use_noise: Whether to add noise
            
        Returns:
            Augmented text
        """
        result = text
        
        if use_synonyms:
            result = self.synonym_replacement(result)
        
        if use_paraphrase:
            result = self.paraphrase_text(result)
        
        if use_noise:
            result = self.add_noise(result)
        
        return result
    
    def augment_dataset(self, df: pd.DataFrame,
                       text_col: str = 'text',
                       label_col: str = 'label',
                       augmentation_factor: float = 2.0,
                       balance_classes: bool = True) -> pd.DataFrame:
        """
        Augment entire dataset with balanced classes
        
        Args:
            df: Input DataFrame
            text_col: Name of text column
            label_col: Name of label column
            augmentation_factor: Factor by which to increase dataset size
            balance_classes: Whether to balance class distribution
            
        Returns:
            Augmented DataFrame
        """
        original_size = len(df)
        target_size = int(original_size * augmentation_factor)
        
        if balance_classes:
            # Balance classes first
            class_counts = df[label_col].value_counts()
            min_class = class_counts.idxmin()
            max_class = class_counts.idxmax()
            
            # Oversample minority class
            minority_df = df[df[label_col] == min_class]
            majority_df = df[df[label_col] == max_class]
            
            # Calculate how many samples to generate
            samples_needed = len(majority_df) - len(minority_df)
            
            if samples_needed > 0:
                # Generate augmented samples for minority class
                augmented_samples = []
                
                for _ in range(samples_needed):
                    # Sample random text from minority class
                    sample_text = minority_df[text_col].sample(1).iloc[0]
                    augmented_text = self.augment_text(sample_text)
                    augmented_samples.append({
                        text_col: augmented_text,
                        label_col: min_class
                    })
                
                # Add augmented samples
                augmented_df = pd.DataFrame(augmented_samples)
                df = pd.concat([df, augmented_df], ignore_index=True)
        
        # Additional augmentation to reach target size
        current_size = len(df)
        additional_samples_needed = max(0, target_size - current_size)
        
        if additional_samples_needed > 0:
            additional_samples = []
            
            for _ in range(additional_samples_needed):
                # Sample random text from entire dataset
                sample_row = df.sample(1).iloc[0]
                augmented_text = self.augment_text(sample_row[text_col])
                additional_samples.append({
                    text_col: augmented_text,
                    label_col: sample_row[label_col]
                })
            
            # Add additional samples
            additional_df = pd.DataFrame(additional_samples)
            df = pd.concat([df, additional_df], ignore_index=True)
        
        # Shuffle final dataset
        df = df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        logger.info(f"Dataset augmented from {original_size} to {len(df)} samples")
        logger.info(f"Final class distribution:\n{df[label_col].value_counts()}")
        
        return df


class DatasetTransformer:
    """
    Main class that combines all transformation utilities
    """
    
    def __init__(self, random_state: int = 42):
        self.converter = StructuredToTextConverter(random_state)
        self.generator = SyntheticTextGenerator(random_state)
        self.augmentor = DataAugmentor(random_state)
        self.random_state = random_state
        
        logger.info("DatasetTransformer initialized")
    
    def transform_outbreak_data(self, input_path: str,
                               output_path: Optional[str] = None,
                               add_synthetic: bool = True,
                               synthetic_ratio: float = 0.3,
                               augment_data: bool = True,
                               augmentation_factor: float = 1.5) -> pd.DataFrame:
        """
        Complete transformation pipeline for outbreak data
        
        Args:
            input_path: Path to structured outbreak data CSV
            output_path: Path to save transformed data (optional)
            add_synthetic: Whether to add synthetic samples
            synthetic_ratio: Ratio of synthetic to real samples
            augment_data: Whether to apply data augmentation
            augmentation_factor: Factor for data augmentation
            
        Returns:
            Transformed DataFrame ready for NLP training
        """
        # Load structured data
        logger.info(f"Loading structured data from {input_path}")
        df = pd.read_csv(input_path)
        
        # Convert to text format
        text_df = self.converter.convert_dataframe_to_text(df)
        
        # Add synthetic negative samples for balance
        n_real_samples = len(text_df)
        n_synthetic = int(n_real_samples * synthetic_ratio)
        
        if add_synthetic:
            synthetic_df = self.generator.generate_balanced_dataset(
                n_samples=n_synthetic,
                positive_ratio=0.3  # More negative samples for balance
            )
            
            # Combine real and synthetic data
            combined_df = pd.concat([text_df, synthetic_df], ignore_index=True)
        else:
            combined_df = text_df
        
        # Apply data augmentation
        if augment_data:
            final_df = self.augmentor.augment_dataset(
                combined_df,
                augmentation_factor=augmentation_factor,
                balance_classes=True
            )
        else:
            final_df = combined_df
        
        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            final_df.to_csv(output_path, index=False)
            logger.info(f"Transformed data saved to {output_path}")
        
        # Log final statistics
        logger.info("Transformation completed:")
        logger.info(f"  Total samples: {len(final_df)}")
        logger.info(f"  Class distribution:\n{final_df['label'].value_counts()}")
        
        return final_df
    
    def create_training_dataset(self, outbreak_data_path: str,
                               output_dir: str,
                               train_size: float = 0.7,
                               val_size: float = 0.15,
                               test_size: float = 0.15) -> Dict[str, pd.DataFrame]:
        """
        Create complete training dataset with splits
        
        Args:
            outbreak_data_path: Path to outbreak data
            output_dir: Directory to save datasets
            train_size: Training set proportion
            val_size: Validation set proportion
            test_size: Test set proportion
            
        Returns:
            Dictionary with train, val, test DataFrames
        """
        # Transform data
        df = self.transform_outbreak_data(outbreak_data_path)
        
        # Create splits
        from sklearn.model_selection import train_test_split
        
        # First split: train+val and test
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=self.random_state,
            stratify=df['label']
        )
        
        # Second split: train and val
        val_ratio = val_size / (train_size + val_size)
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_ratio, random_state=self.random_state,
            stratify=train_val_df['label']
        )
        
        # Save splits
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        splits = {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
        
        for split_name, split_df in splits.items():
            file_path = output_path / f"{split_name}.csv"
            split_df.to_csv(file_path, index=False)
            logger.info(f"Saved {split_name} split: {len(split_df)} samples to {file_path}")
        
        return splits


if __name__ == "__main__":
    print("Dataset Transformation Utilities")
    print("=" * 50)
    
    # Initialize transformer
    transformer = DatasetTransformer()
    
    # Example usage with sample data
    input_file = "data/raw/disease_outbreaks_minimal.csv"
    output_dir = "data/processed/transformed"
    
    try:
        # Create training dataset
        splits = transformer.create_training_dataset(
            outbreak_data_path=input_file,
            output_dir=output_dir
        )
        
        print("\nDataset transformation completed successfully!")
        print(f"Train samples: {len(splits['train'])}")
        print(f"Validation samples: {len(splits['val'])}")
        print(f"Test samples: {len(splits['test'])}")
        
        # Show sample texts
        print("\nSample transformed texts:")
        for i, text in enumerate(splits['train']['text'].head(3)):
            label = splits['train']['label'].iloc[i]
            print(f"{i+1}. [{label}] {text}")
        
    except Exception as e:
        print(f"Error: {e}")