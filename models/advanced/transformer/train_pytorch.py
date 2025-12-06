import torch
import torch.nn as nn
import torch.optim as optim
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        return x + self.pe[:x.size(0), :]

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, num_classes, max_len=100):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)

        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        self.fc = nn.Linear(d_model, num_classes)
        self.d_model = d_model

    def forward(self, src):
        # src: (batch_size, seq_len) -> Transpose to (seq_len, batch_size)
        src = src.transpose(0, 1)
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        # Average pooling across sequence
        output = output.mean(dim=0)
        return self.fc(output)

def train():
    print("Training Transformer with PyTorch (Dummy Text Classification)...")

    # Hyperparameters
    VOCAB_SIZE = 1000
    D_MODEL = 64
    NHEAD = 4
    NUM_LAYERS = 2
    NUM_CLASSES = 2
    MAX_LEN = 20
    BATCH_SIZE = 32

    # Dummy Data: Random sequences of integers
    X_train = torch.randint(0, VOCAB_SIZE, (100, MAX_LEN))
    y_train = torch.randint(0, NUM_CLASSES, (100,))

    model = TransformerClassifier(VOCAB_SIZE, D_MODEL, NHEAD, NUM_LAYERS, NUM_CLASSES, MAX_LEN)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 2 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    print("PyTorch Transformer Training Complete.")

if __name__ == "__main__":
    train()
