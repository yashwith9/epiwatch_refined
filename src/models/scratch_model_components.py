"""
Custom Neural Network Components for Epidemic Detection (Built from Scratch)
This module implements modular components for a scratch-built LSTM model with attention mechanism.

Components:
- VocabularyManager: Handles vocabulary creation and management
- EmbeddingLayer: Custom embedding layer with vocabulary management
- BiLSTMEncoder: Bidirectional LSTM encoder
- AttentionMechanism: Attention mechanism for focusing on important text parts
- ClassificationHead: Final classification layers
- ScratchEpidemicModel: Complete model combining all components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
import pickle
import os
from collections import Counter


class VocabularyManager:
    """
    Manages vocabulary creation, encoding, and decoding for text data.
    Handles special tokens and provides utilities for text-to-index conversion.
    """
    
    def __init__(self, min_freq: int = 2, max_vocab_size: Optional[int] = None):
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.vocab = {}
        self.idx_to_word = {}
        self.word_counts = Counter()
        
        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.START_TOKEN = '<START>'
        self.END_TOKEN = '<END>'
        
        # Initialize with special tokens
        self._init_special_tokens()
    
    def _init_special_tokens(self):
        """Initialize vocabulary with special tokens"""
        special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.START_TOKEN, self.END_TOKEN]
        for i, token in enumerate(special_tokens):
            self.vocab[token] = i
            self.idx_to_word[i] = token
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts
        
        Args:
            texts: List of text strings to build vocabulary from
        """
        print("Building vocabulary...")
        
        # Count word frequencies
        for text in texts:
            words = text.lower().split()
            self.word_counts.update(words)
        
        # Add words to vocabulary based on frequency
        vocab_idx = len(self.vocab)  # Start after special tokens
        
        # Sort by frequency (most frequent first)
        sorted_words = self.word_counts.most_common()
        
        for word, count in sorted_words:
            if count >= self.min_freq and word not in self.vocab:
                if self.max_vocab_size and len(self.vocab) >= self.max_vocab_size:
                    break
                
                self.vocab[word] = vocab_idx
                self.idx_to_word[vocab_idx] = word
                vocab_idx += 1
        
        print(f"Vocabulary built with {len(self.vocab)} tokens")
        print(f"Most frequent words: {list(dict(sorted_words[:10]).keys())}")
    
    def encode_text(self, text: str) -> List[int]:
        """
        Convert text to list of token indices
        
        Args:
            text: Input text string
            
        Returns:
            List of token indices
        """
        words = text.lower().split()
        return [self.vocab.get(word, self.vocab[self.UNK_TOKEN]) for word in words]
    
    def decode_indices(self, indices: List[int]) -> str:
        """
        Convert list of indices back to text
        
        Args:
            indices: List of token indices
            
        Returns:
            Decoded text string
        """
        words = [self.idx_to_word.get(idx, self.UNK_TOKEN) for idx in indices]
        return ' '.join(words)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.vocab)
    
    def get_pad_idx(self) -> int:
        """Get padding token index"""
        return self.vocab[self.PAD_TOKEN]
    
    def get_unk_idx(self) -> int:
        """Get unknown token index"""
        return self.vocab[self.UNK_TOKEN]
    
    def save_vocab(self, filepath: str) -> None:
        """Save vocabulary to file"""
        vocab_data = {
            'vocab': self.vocab,
            'idx_to_word': self.idx_to_word,
            'word_counts': dict(self.word_counts),
            'min_freq': self.min_freq,
            'max_vocab_size': self.max_vocab_size
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(vocab_data, f)
        print(f"Vocabulary saved to {filepath}")
    
    def load_vocab(self, filepath: str) -> None:
        """Load vocabulary from file"""
        with open(filepath, 'rb') as f:
            vocab_data = pickle.load(f)
        
        self.vocab = vocab_data['vocab']
        self.idx_to_word = vocab_data['idx_to_word']
        self.word_counts = Counter(vocab_data['word_counts'])
        self.min_freq = vocab_data['min_freq']
        self.max_vocab_size = vocab_data['max_vocab_size']
        print(f"Vocabulary loaded from {filepath}")


class EmbeddingLayer(nn.Module):
    """
    Custom embedding layer with vocabulary management and optional pre-trained embeddings
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, padding_idx: int = 0, 
                 dropout: float = 0.1, pretrained_embeddings: Optional[torch.Tensor] = None):
        super(EmbeddingLayer, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        
        # Create embedding layer
        self.embedding = nn.Embedding(
            vocab_size, 
            embedding_dim, 
            padding_idx=padding_idx
        )
        
        # Initialize with pre-trained embeddings if provided
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            print("Initialized with pre-trained embeddings")
        else:
            # Xavier uniform initialization
            nn.init.xavier_uniform_(self.embedding.weight)
            # Set padding embedding to zero
            with torch.no_grad():
                self.embedding.weight[padding_idx].fill_(0)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through embedding layer
        
        Args:
            x: Input tensor of token indices [batch_size, seq_len]
            
        Returns:
            Embedded representations [batch_size, seq_len, embedding_dim]
        """
        # Get embeddings
        embedded = self.embedding(x)
        
        # Apply layer normalization and dropout
        embedded = self.layer_norm(embedded)
        embedded = self.dropout(embedded)
        
        return embedded


class BiLSTMEncoder(nn.Module):
    """
    Bidirectional LSTM encoder with multiple layers and dropout
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, 
                 dropout: float = 0.3, bidirectional: bool = True):
        super(BiLSTMEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output dimension
        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through BiLSTM encoder
        
        Args:
            x: Input embeddings [batch_size, seq_len, input_dim]
            lengths: Actual sequence lengths for packing (optional)
            
        Returns:
            output: LSTM outputs [batch_size, seq_len, hidden_dim * 2]
            hidden_state: Final hidden and cell states
        """
        batch_size = x.size(0)
        
        # Pack sequences if lengths provided
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        # LSTM forward pass
        output, (hidden, cell) = self.lstm(x)
        
        # Unpack sequences if they were packed
        if lengths is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        
        # Apply dropout
        output = self.dropout(output)
        
        return output, (hidden, cell)


class AttentionMechanism(nn.Module):
    """
    Attention mechanism to focus on important parts of the sequence
    Implements both additive (Bahdanau) and multiplicative (Luong) attention
    """
    
    def __init__(self, hidden_dim: int, attention_type: str = 'additive'):
        super(AttentionMechanism, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.attention_type = attention_type
        
        if attention_type == 'additive':
            # Additive attention (Bahdanau)
            self.W_a = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.U_a = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.v_a = nn.Linear(hidden_dim, 1, bias=False)
        elif attention_type == 'multiplicative':
            # Multiplicative attention (Luong)
            self.W_a = nn.Linear(hidden_dim, hidden_dim, bias=False)
        else:
            raise ValueError("attention_type must be 'additive' or 'multiplicative'")
        
        # Dropout for attention weights
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, encoder_outputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention weights and context vector
        
        Args:
            encoder_outputs: LSTM outputs [batch_size, seq_len, hidden_dim]
            mask: Padding mask [batch_size, seq_len] (optional)
            
        Returns:
            context_vector: Weighted sum of encoder outputs [batch_size, hidden_dim]
            attention_weights: Attention weights [batch_size, seq_len]
        """
        batch_size, seq_len, hidden_dim = encoder_outputs.size()
        
        if self.attention_type == 'additive':
            # Additive attention
            # Use mean of encoder outputs as query
            query = encoder_outputs.mean(dim=1, keepdim=True)  # [batch_size, 1, hidden_dim]
            
            # Compute attention scores
            scores = self.v_a(torch.tanh(
                self.W_a(encoder_outputs) + self.U_a(query)
            )).squeeze(-1)  # [batch_size, seq_len]
            
        elif self.attention_type == 'multiplicative':
            # Multiplicative attention
            # Use mean of encoder outputs as query
            query = encoder_outputs.mean(dim=1, keepdim=True)  # [batch_size, 1, hidden_dim]
            
            # Compute attention scores
            transformed_outputs = self.W_a(encoder_outputs)  # [batch_size, seq_len, hidden_dim]
            scores = torch.bmm(query, transformed_outputs.transpose(1, 2)).squeeze(1)  # [batch_size, seq_len]
        
        # Apply mask if provided
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        
        # Compute attention weights
        attention_weights = F.softmax(scores, dim=1)  # [batch_size, seq_len]
        attention_weights = self.dropout(attention_weights)
        
        # Compute context vector
        context_vector = torch.bmm(
            attention_weights.unsqueeze(1), encoder_outputs
        ).squeeze(1)  # [batch_size, hidden_dim]
        
        return context_vector, attention_weights


class ClassificationHead(nn.Module):
    """
    Classification head with multiple fully connected layers and regularization
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], num_classes: int = 1, 
                 dropout: float = 0.3, use_batch_norm: bool = True):
        super(ClassificationHead, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.use_batch_norm = use_batch_norm
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            layers.append(nn.ReLU())
            
            # Dropout
            layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Final classification layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        # For binary classification, add sigmoid
        if num_classes == 1:
            layers.append(nn.Sigmoid())
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through classification head
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Classification logits/probabilities [batch_size, num_classes]
        """
        return self.classifier(x)


class ScratchEpidemicModel(nn.Module):
    """
    Complete scratch-built model for epidemic detection combining all components
    Architecture: Embedding -> BiLSTM -> Attention -> Classification
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 256, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.3, attention_type: str = 'additive',
                 classifier_hidden_dims: List[int] = [64, 32], padding_idx: int = 0):
        super(ScratchEpidemicModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.padding_idx = padding_idx
        
        # Model components
        self.embedding = EmbeddingLayer(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            dropout=dropout
        )
        
        self.encoder = BiLSTMEncoder(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True
        )
        
        self.attention = AttentionMechanism(
            hidden_dim=hidden_dim * 2,  # BiLSTM output dimension
            attention_type=attention_type
        )
        
        self.classifier = ClassificationHead(
            input_dim=hidden_dim * 2,
            hidden_dims=classifier_hidden_dims,
            num_classes=1,  # Binary classification
            dropout=dropout
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) > 1:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def create_padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create padding mask for attention mechanism
        
        Args:
            x: Input tensor [batch_size, seq_len]
            
        Returns:
            Mask tensor [batch_size, seq_len] (1 for valid tokens, 0 for padding)
        """
        return (x != self.padding_idx).float()
    
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        """
        Forward pass through the complete model
        
        Args:
            x: Input token indices [batch_size, seq_len]
            return_attention: Whether to return attention weights
            
        Returns:
            predictions: Model predictions [batch_size, 1]
            attention_weights: Attention weights (if return_attention=True)
        """
        # Create padding mask
        mask = self.create_padding_mask(x)
        
        # Embedding layer
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        
        # BiLSTM encoder
        encoder_outputs, _ = self.encoder(embedded)  # [batch_size, seq_len, hidden_dim * 2]
        
        # Attention mechanism
        context_vector, attention_weights = self.attention(encoder_outputs, mask)
        
        # Classification head
        predictions = self.classifier(context_vector)  # [batch_size, 1]
        
        if return_attention:
            return predictions.squeeze(-1), attention_weights
        else:
            return predictions.squeeze(-1)
    
    def get_model_info(self) -> Dict:
        """Get model architecture information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }


if __name__ == "__main__":
    # Example usage and testing
    print("Scratch Model Components for Epidemic Detection")
    print("=" * 60)
    
    # Test vocabulary manager
    sample_texts = [
        "outbreak detected in region",
        "epidemic spreading rapidly",
        "health officials confirm cases",
        "disease outbreak reported"
    ]
    
    vocab_manager = VocabularyManager(min_freq=1)
    vocab_manager.build_vocab(sample_texts)
    
    print(f"\nVocabulary size: {vocab_manager.get_vocab_size()}")
    print(f"Sample encoding: {vocab_manager.encode_text('outbreak detected')}")
    
    # Test model
    model = ScratchEpidemicModel(
        vocab_size=vocab_manager.get_vocab_size(),
        embedding_dim=128,
        hidden_dim=64,
        num_layers=2,
        padding_idx=vocab_manager.get_pad_idx()
    )
    
    print(f"\nModel Info:")
    info = model.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test forward pass
    sample_input = torch.randint(0, vocab_manager.get_vocab_size(), (2, 10))
    with torch.no_grad():
        output, attention = model(sample_input, return_attention=True)
        print(f"\nSample output shape: {output.shape}")
        print(f"Attention weights shape: {attention.shape}")