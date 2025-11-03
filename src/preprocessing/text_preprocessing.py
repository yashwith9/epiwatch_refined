"""
Text Preprocessing Pipeline for Multilingual Epidemic Detection
Language detection, translation, and text cleaning
"""

import re
import string
from typing import List, Dict
import pandas as pd
import numpy as np

try:
    from langdetect import detect, detect_langs
    from langdetect.lang_detect_exception import LangDetectException
except ImportError:
    print("Warning: langdetect not installed. Install with: pip install langdetect")

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
except ImportError:
    print("Warning: nltk not installed. Install with: pip install nltk")


class TextPreprocessor:
    """
    Comprehensive text preprocessing for epidemic detection
    """
    
    def __init__(self, target_language='en', use_translation=False):
        """
        Args:
            target_language: Target language for translation (default: 'en')
            use_translation: Whether to translate non-English text
        """
        self.target_language = target_language
        self.use_translation = use_translation
        
        # Download NLTK data if needed
        try:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
        except:
            print("Downloading NLTK data...")
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
        
        # Epidemic-related keywords (keep these even if they're in stopwords)
        self.epidemic_keywords = {
            'outbreak', 'epidemic', 'pandemic', 'disease', 'infection',
            'virus', 'bacteria', 'fever', 'cases', 'death', 'deaths',
            'sick', 'ill', 'hospital', 'symptoms', 'spread', 'transmission',
            'quarantine', 'isolation', 'vaccine', 'treatment', 'covid',
            'influenza', 'dengue', 'malaria', 'cholera', 'ebola', 'measles',
            'confirmed', 'suspected', 'reported', 'health', 'alert'
        }
    
    def detect_language(self, text: str) -> str:
        """
        Detect language of text
        
        Args:
            text: Input text
            
        Returns:
            ISO 639-1 language code (e.g., 'en', 'hi', 'es')
        """
        try:
            lang = detect(text)
            return lang
        except LangDetectException:
            return 'unknown'
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove numbers (but keep if part of disease name like COVID-19)
        # text = re.sub(r'\b\d+\b', '', text)
        
        return text
    
    def remove_stopwords(self, text: str) -> str:
        """
        Remove stopwords while preserving epidemic keywords
        
        Args:
            text: Input text
            
        Returns:
            Text with stopwords removed
        """
        words = text.split()
        
        # Remove stopwords but keep epidemic keywords
        filtered_words = [
            word for word in words 
            if word not in self.stop_words or word in self.epidemic_keywords
        ]
        
        return ' '.join(filtered_words)
    
    def lemmatize_text(self, text: str) -> str:
        """
        Lemmatize words in text
        
        Args:
            text: Input text
            
        Returns:
            Lemmatized text
        """
        words = text.split()
        lemmatized = [self.lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized)
    
    def remove_punctuation(self, text: str) -> str:
        """
        Remove punctuation from text
        
        Args:
            text: Input text
            
        Returns:
            Text without punctuation
        """
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def preprocess(self, text: str, 
                  remove_stop=True, 
                  lemmatize=True, 
                  remove_punct=True) -> str:
        """
        Complete preprocessing pipeline
        
        Args:
            text: Input text
            remove_stop: Whether to remove stopwords
            lemmatize: Whether to lemmatize
            remove_punct: Whether to remove punctuation
            
        Returns:
            Preprocessed text
        """
        # Clean
        text = self.clean_text(text)
        
        # Remove punctuation
        if remove_punct:
            text = self.remove_punctuation(text)
        
        # Remove stopwords
        if remove_stop:
            text = self.remove_stopwords(text)
        
        # Lemmatize
        if lemmatize:
            text = self.lemmatize_text(text)
        
        return text
    
    def preprocess_dataset(self, texts: List[str], 
                          show_progress=True) -> List[str]:
        """
        Preprocess a list of texts
        
        Args:
            texts: List of input texts
            show_progress: Whether to show progress bar
            
        Returns:
            List of preprocessed texts
        """
        if show_progress:
            try:
                from tqdm import tqdm
                texts = tqdm(texts, desc="Preprocessing")
            except:
                pass
        
        preprocessed = [self.preprocess(text) for text in texts]
        return preprocessed
    
    def extract_features(self, text: str) -> Dict:
        """
        Extract epidemic-relevant features from text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of features
        """
        text_lower = text.lower()
        
        features = {
            'length': len(text),
            'word_count': len(text.split()),
            'has_numbers': bool(re.search(r'\d', text)),
            'has_location': bool(re.search(r'\b(city|region|country|area|district|province)\b', text_lower)),
            'has_date': bool(re.search(r'\b(today|yesterday|week|month|year|\d{1,2}/\d{1,2})\b', text_lower)),
            'has_severity': bool(re.search(r'\b(severe|critical|mild|moderate|serious)\b', text_lower)),
            'has_symptoms': bool(re.search(r'\b(fever|cough|pain|symptom|sick|ill)\b', text_lower)),
            'has_death': bool(re.search(r'\b(death|died|fatal|mortality)\b', text_lower)),
            'epidemic_keyword_count': sum(1 for keyword in self.epidemic_keywords if keyword in text_lower)
        }
        
        return features


class DatasetBuilder:
    """
    Build training dataset from raw data
    """
    
    def __init__(self, preprocessor=None):
        self.preprocessor = preprocessor or TextPreprocessor()
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load data from file (CSV, JSON, etc.)
        
        Args:
            filepath: Path to data file
            
        Returns:
            DataFrame with loaded data
        """
        if filepath.endswith('.csv'):
            return pd.read_csv(filepath)
        elif filepath.endswith('.json'):
            return pd.read_json(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
    
    def balance_dataset(self, df: pd.DataFrame, 
                       text_col='text', 
                       label_col='label',
                       strategy='undersample') -> pd.DataFrame:
        """
        Balance dataset classes
        
        Args:
            df: Input DataFrame
            text_col: Name of text column
            label_col: Name of label column
            strategy: 'undersample' or 'oversample'
            
        Returns:
            Balanced DataFrame
        """
        class_counts = df[label_col].value_counts()
        print(f"Original class distribution:\n{class_counts}\n")
        
        if strategy == 'undersample':
            # Undersample majority class
            min_count = class_counts.min()
            balanced_df = df.groupby(label_col).apply(
                lambda x: x.sample(min_count, random_state=42)
            ).reset_index(drop=True)
            
        elif strategy == 'oversample':
            # Oversample minority class
            max_count = class_counts.max()
            balanced_df = df.groupby(label_col).apply(
                lambda x: x.sample(max_count, replace=True, random_state=42)
            ).reset_index(drop=True)
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        print(f"Balanced class distribution:\n{balanced_df[label_col].value_counts()}\n")
        return balanced_df
    
    def prepare_train_test_split(self, df: pd.DataFrame,
                                text_col='text',
                                label_col='label',
                                test_size=0.2,
                                val_size=0.1) -> Dict:
        """
        Split dataset into train, validation, and test sets
        
        Args:
            df: Input DataFrame
            text_col: Name of text column
            label_col: Name of label column
            test_size: Proportion of test set
            val_size: Proportion of validation set
            
        Returns:
            Dictionary with train, val, test splits
        """
        from sklearn.model_selection import train_test_split
        
        # First split: train+val and test
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=42, stratify=df[label_col]
        )
        
        # Second split: train and val
        val_ratio = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_ratio, random_state=42, stratify=train_val_df[label_col]
        )
        
        print(f"Dataset split:")
        print(f"  Train: {len(train_df)} samples")
        print(f"  Validation: {len(val_df)} samples")
        print(f"  Test: {len(test_df)} samples")
        
        return {
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
    
    def create_sample_dataset(self, n_samples=1000, save_path=None):
        """
        Create a sample dataset for testing
        
        Args:
            n_samples: Number of samples to generate
            save_path: Path to save dataset (optional)
            
        Returns:
            DataFrame with sample data
        """
        # Outbreak-related texts
        outbreak_texts = [
            "Multiple cases of dengue fever reported in Mumbai region",
            "Health officials confirm cholera outbreak in rural district",
            "COVID-19 cases surge in Delhi hospitals",
            "Malaria outbreak declared in affected villages",
            "Influenza spreading rapidly across schools",
            "Health alert issued for suspected measles cases",
            "Hospital admits 50 patients with fever symptoms",
            "Disease surveillance detects unusual illness pattern",
            "Public health emergency declared in coastal areas",
            "Epidemic response team deployed to investigate"
        ]
        
        # Non-outbreak texts
        normal_texts = [
            "Local community celebrates health awareness week",
            "New medical facility inaugurated in city center",
            "Health minister announces vaccination drive",
            "Sports event promotes fitness and wellness",
            "Hospital celebrates anniversary with free checkups",
            "Medical research breakthrough announced",
            "Healthcare workers receive training on new protocols",
            "Public health campaign focuses on nutrition",
            "Government invests in healthcare infrastructure",
            "Medical college announces new admissions"
        ]
        
        # Generate samples
        texts = []
        labels = []
        
        for _ in range(n_samples // 2):
            texts.append(np.random.choice(outbreak_texts))
            labels.append(1)
            
            texts.append(np.random.choice(normal_texts))
            labels.append(0)
        
        df = pd.DataFrame({
            'text': texts,
            'label': labels,
            'region': np.random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Chennai'], n_samples),
            'disease': np.random.choice(['Dengue', 'COVID-19', 'Malaria', 'Influenza'], n_samples),
            'date': pd.date_range(start='2024-01-01', periods=n_samples, freq='H')
        })
        
        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        if save_path:
            df.to_csv(save_path, index=False)
            print(f"✓ Sample dataset saved: {save_path}")
        
        return df


if __name__ == "__main__":
    print("Text Preprocessing Pipeline for EpiWatch")
    print("=" * 60)
    
    # Example usage
    preprocessor = TextPreprocessor()
    
    example_text = "Breaking: Multiple cases of Dengue fever reported in Mumbai! Health officials are investigating. #HealthAlert"
    
    print(f"\nOriginal text:\n{example_text}")
    print(f"\nCleaned text:\n{preprocessor.clean_text(example_text)}")
    print(f"\nPreprocessed text:\n{preprocessor.preprocess(example_text)}")
    print(f"\nLanguage: {preprocessor.detect_language(example_text)}")
    print(f"\nFeatures: {preprocessor.extract_features(example_text)}")
