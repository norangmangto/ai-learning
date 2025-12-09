"""
RNN-based Text Classification with LSTM and GRU
Alternative to Transformer-based approaches
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def train():
    print("Training Text Classification with LSTM...")

    # 1. Prepare Data
    try:
        categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
        newsgroups = fetch_20newsgroups(subset='train', categories=categories,
                                       remove=('headers', 'footers', 'quotes'))
        train_texts = newsgroups.data[:500]
        train_labels = newsgroups.target[:500]

        test_newsgroups = fetch_20newsgroups(subset='test', categories=categories,
                                            remove=('headers', 'footers', 'quotes'))
        test_texts = test_newsgroups.data[:100]
        test_labels = test_newsgroups.target[:100]
    except:
        print("Warning: Could not load data. Using synthetic...")
        train_texts, train_labels = create_synthetic_data(500)
        test_texts, test_labels = create_synthetic_data(100)

    # 2. Build Vocabulary
    vectorizer = CountVectorizer(max_features=1000)
    vectorizer.fit(train_texts)

    vocab_size = len(vectorizer.vocabulary_)
    print(f"Vocabulary size: {vocab_size}")

    # 3. Convert texts to sequences
    def texts_to_sequences(texts, vectorizer, max_len=100):
        sequences = []
        for text in texts:
            vec = vectorizer.transform([text]).toarray()[0]
            indices = np.nonzero(vec)[0][:max_len]
            sequences.append(indices)
        return sequences

    train_sequences = texts_to_sequences(train_texts, vectorizer)
    test_sequences = texts_to_sequences(test_texts, vectorizer)

    # 4. Define LSTM Model
    class LSTMClassifier(nn.Module):
        def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_classes=4):
            super(LSTMClassifier, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2,
                               batch_first=True, dropout=0.3, bidirectional=True)
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim * 2, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )

        def forward(self, x):
            x = self.embedding(x)
            _, (hidden, _) = self.lstm(x)
            # Concatenate forward and backward hidden states
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
            output = self.fc(hidden)
            return output

    # 5. Prepare DataLoader
    class TextDataset(Dataset):
        def __init__(self, sequences, labels, vocab_size=1000, max_len=100):
            self.sequences = sequences
            self.labels = labels
            self.vocab_size = vocab_size
            self.max_len = max_len

        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, idx):
            seq = self.sequences[idx]
            # Pad or truncate to max_len
            seq = seq[:self.max_len]
            seq = np.pad(seq, (0, self.max_len - len(seq)), 'constant')
            return torch.tensor(seq, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

    train_dataset = TextDataset(train_sequences, train_labels, vocab_size)
    test_dataset = TextDataset(test_sequences, test_labels, vocab_size)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 6. Train Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMClassifier(vocab_size)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    print("\nTraining LSTM...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for sequences, labels in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 2 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

    # 7. Evaluate
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"\nLSTM Text Classification Accuracy: {accuracy:.4f}")

    # 8. QA Validation
    print("\n=== QA Validation ===")
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    print(f"F1-Score (weighted): {f1:.4f}")

    print("\n--- Sanity Checks ---")
    if accuracy >= 0.6:
        print(f"✓ Good accuracy: {accuracy:.4f}")
    else:
        print(f"⚠ Moderate accuracy: {accuracy:.4f}")

    print("\n=== Overall Validation Result ===")
    validation_passed = accuracy >= 0.55

    if validation_passed:
        print("✓ Validation PASSED")
    else:
        print("✗ Validation FAILED")

    return model


def create_synthetic_data(size):
    """Create synthetic text classification data"""
    categories = {
        0: ["math physics science quantum", "calculus integration derivative"],
        1: ["religion faith spiritual belief", "church prayer worship"],
        2: ["graphics image visual computer", "rendering pixels shader"],
        3: ["medicine health doctor treatment", "disease diagnosis cure"],
    }

    texts = []
    labels = []

    for label in range(4):
        for _ in range(size // 4):
            text = " ".join(np.random.choice(categories[label], 3))
            texts.append(text)
            labels.append(label)

    return texts[:size], np.array(labels[:size])


if __name__ == "__main__":
    train()
