"""
Custom Neural Network for Epidemic Detection (Built from Scratch)
This model uses LSTM + Attention mechanism for binary classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np


class EpidemicDataset(Dataset):
    """Custom Dataset for epidemic detection"""
    
    def __init__(self, texts, labels, vocab, max_length=256):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Convert text to indices
        indices = [self.vocab.get(word, self.vocab['<UNK>']) 
                   for word in text.lower().split()]
        
        # Pad or truncate
        if len(indices) < self.max_length:
            indices += [self.vocab['<PAD>']] * (self.max_length - len(indices))
        else:
            indices = indices[:self.max_length]
            
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.float)


class AttentionLayer(nn.Module):
    """Attention mechanism to focus on important parts of text"""
    
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, lstm_output):
        # lstm_output shape: (batch, seq_len, hidden_dim * 2)
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        # attention_weights shape: (batch, seq_len, 1)
        
        # Apply attention weights
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        # context_vector shape: (batch, hidden_dim * 2)
        
        return context_vector, attention_weights


class CustomEpiDetector(nn.Module):
    """
    Custom Neural Network for Epidemic Detection
    Architecture: Embedding -> Bi-LSTM -> Attention -> Dense -> Sigmoid
    """
    
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=128, 
                 num_layers=2, dropout=0.3):
        super(CustomEpiDetector, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention layer
        self.attention = AttentionLayer(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)
        
    def forward(self, x):
        # x shape: (batch, seq_len)
        
        # Embedding
        embedded = self.embedding(x)
        # embedded shape: (batch, seq_len, embedding_dim)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # lstm_out shape: (batch, seq_len, hidden_dim * 2)
        
        # Attention
        context_vector, attention_weights = self.attention(lstm_out)
        # context_vector shape: (batch, hidden_dim * 2)
        
        # Fully connected layers with dropout and batch norm
        x = self.dropout(context_vector)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        
        return x.squeeze(1), attention_weights


class ModelTrainer:
    """Trainer class for custom model"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_epoch(self, train_loader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for texts, labels in train_loader:
            texts, labels = texts.to(self.device), labels.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs, _ = self.model(texts)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader, criterion):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for texts, labels in val_loader:
                texts, labels = texts.to(self.device), labels.to(self.device)
                
                outputs, _ = self.model(texts)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                predicted = (outputs > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs=10, lr=0.001):
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=2, factor=0.5
        )
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print("-" * 50)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'models/saved/custom_best.pt')
        
        return self.train_losses, self.val_losses
    
    def predict(self, test_loader):
        """Generate predictions for test data"""
        self.model.eval()
        predictions = []
        attention_weights_list = []
        
        with torch.no_grad():
            for texts, _ in test_loader:
                texts = texts.to(self.device)
                outputs, attention_weights = self.model(texts)
                predictions.extend(outputs.cpu().numpy())
                attention_weights_list.append(attention_weights.cpu().numpy())
        
        return np.array(predictions), attention_weights_list


def build_vocab(texts, min_freq=2):
    """Build vocabulary from texts"""
    word_freq = {}
    for text in texts:
        for word in text.lower().split():
            word_freq[word] = word_freq.get(word, 0) + 1
    
    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    for word, freq in word_freq.items():
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1
    
    return vocab


if __name__ == "__main__":
    # Example usage
    print("Custom Epidemic Detector Model")
    print("=" * 50)
    
    # Initialize model
    vocab_size = 10000
    model = CustomEpiDetector(
        vocab_size=vocab_size,
        embedding_dim=256,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3
    )
    
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(model)
