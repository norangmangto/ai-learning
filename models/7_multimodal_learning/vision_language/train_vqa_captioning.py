"""
Vision-Language Model with PyTorch

Vision-Language models for visual question answering and image captioning:
- Visual Question Answering (VQA): answer questions about images
- Image Captioning: generate natural language descriptions
- Cross-modal attention between vision and language
- Applications: assistive technology, content understanding, robotics

This implementation includes:
- CNN image encoder
- LSTM text encoder/decoder
- Cross-modal attention
- Visual Question Answering
- Image captioning
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import time


class ImageFeatureExtractor(nn.Module):
    """CNN for extracting image features."""
    def __init__(self, feature_dim=512):
        super(ImageFeatureExtractor, self).__init__()

        self.conv_layers = nn.Sequential(
            # 3x64x64 -> 64x32x32
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 64x32x32 -> 128x16x16
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 128x16x16 -> 256x8x8
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 256x8x8 -> 512x4x4
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Don't pool - keep spatial features for attention
        self.projection = nn.Linear(512, feature_dim)

    def forward(self, x):
        # x: (batch, 3, 64, 64)
        x = self.conv_layers(x)  # (batch, 512, 4, 4)

        batch_size, channels, h, w = x.size()

        # Reshape to (batch, num_regions, channels)
        x = x.view(batch_size, channels, h * w)
        x = x.permute(0, 2, 1)  # (batch, 16, 512)

        # Project
        x = self.projection(x)  # (batch, 16, feature_dim)

        return x


class AttentionMechanism(nn.Module):
    """Attention over image regions."""
    def __init__(self, visual_dim, text_dim, hidden_dim):
        super(AttentionMechanism, self).__init__()

        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.attention_proj = nn.Linear(hidden_dim, 1)

    def forward(self, visual_features, text_features):
        """
        visual_features: (batch, num_regions, visual_dim)
        text_features: (batch, text_dim)
        """
        batch_size, num_regions, visual_dim = visual_features.size()

        # Project visual features
        v = self.visual_proj(visual_features)  # (batch, num_regions, hidden_dim)

        # Project text features and expand
        t = self.text_proj(text_features).unsqueeze(1)  # (batch, 1, hidden_dim)
        t = t.expand(-1, num_regions, -1)  # (batch, num_regions, hidden_dim)

        # Compute attention scores
        combined = torch.tanh(v + t)
        scores = self.attention_proj(combined).squeeze(-1)  # (batch, num_regions)

        # Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=1)  # (batch, num_regions)

        # Weighted sum of visual features
        attended_features = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch, 1, num_regions)
            visual_features  # (batch, num_regions, visual_dim)
        ).squeeze(1)  # (batch, visual_dim)

        return attended_features, attention_weights


class VQAModel(nn.Module):
    """
    Visual Question Answering model.

    Given an image and a question, predict an answer.
    """
    def __init__(self, vocab_size, num_answers, embed_dim=256, hidden_dim=512):
        super(VQAModel, self).__init__()

        # Image encoder
        self.image_encoder = ImageFeatureExtractor(embed_dim)

        # Question encoder
        self.word_embedding = nn.Embedding(vocab_size, embed_dim)
        self.question_lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

        # Attention
        self.attention = AttentionMechanism(embed_dim, hidden_dim, hidden_dim)

        # Fusion and classification
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, num_answers)
        )

    def forward(self, images, questions):
        """
        images: (batch, 3, 64, 64)
        questions: (batch, seq_len)
        """
        # Extract image features
        visual_features = self.image_encoder(images)  # (batch, num_regions, embed_dim)

        # Encode question
        q_embedded = self.word_embedding(questions)
        _, (q_hidden, _) = self.question_lstm(q_embedded)
        q_features = q_hidden[-1]  # (batch, hidden_dim)

        # Attend to relevant image regions
        attended_visual, attention_weights = self.attention(visual_features, q_features)

        # Fuse visual and question features
        combined = torch.cat([attended_visual, q_features], dim=1)
        logits = self.fusion(combined)

        return logits, attention_weights


class ImageCaptioningModel(nn.Module):
    """
    Image captioning model with attention.

    Generate a caption describing the image.
    """
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, max_length=20):
        super(ImageCaptioningModel, self).__init__()

        self.vocab_size = vocab_size
        self.max_length = max_length

        # Image encoder
        self.image_encoder = ImageFeatureExtractor(embed_dim)

        # Caption decoder
        self.word_embedding = nn.Embedding(vocab_size, embed_dim)
        self.decoder_lstm = nn.LSTM(embed_dim + embed_dim, hidden_dim, batch_first=True)

        # Attention
        self.attention = AttentionMechanism(embed_dim, hidden_dim, hidden_dim)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, images, captions):
        """
        images: (batch, 3, 64, 64)
        captions: (batch, seq_len) - ground truth captions for training
        """
        batch_size = images.size(0)

        # Extract image features
        visual_features = self.image_encoder(images)  # (batch, num_regions, embed_dim)

        # Decode captions
        caption_embedded = self.word_embedding(captions)  # (batch, seq_len, embed_dim)

        outputs = []
        hidden = None

        for t in range(captions.size(1)):
            # Current word embedding
            word_embed = caption_embedded[:, t, :]  # (batch, embed_dim)

            # Attend to image
            if hidden is None:
                # Initial state
                h = torch.zeros(batch_size, self.decoder_lstm.hidden_size, device=images.device)
            else:
                h = hidden[0][-1]

            attended_visual, _ = self.attention(visual_features, h)

            # Combine word and visual context
            decoder_input = torch.cat([word_embed, attended_visual], dim=1)
            decoder_input = decoder_input.unsqueeze(1)  # (batch, 1, embed_dim*2)

            # LSTM step
            output, hidden = self.decoder_lstm(decoder_input, hidden)

            # Project to vocabulary
            output = self.output_proj(output.squeeze(1))
            outputs.append(output)

        outputs = torch.stack(outputs, dim=1)  # (batch, seq_len, vocab_size)

        return outputs

    @torch.no_grad()
    def generate(self, images, start_token=1, end_token=2, max_length=20):
        """Generate captions autoregressively."""
        self.eval()
        batch_size = images.size(0)
        device = images.device

        # Extract image features
        visual_features = self.image_encoder(images)

        # Start with start token
        input_tokens = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)

        generated = []
        hidden = None

        for _ in range(max_length):
            word_embed = self.word_embedding(input_tokens[:, -1])

            if hidden is None:
                h = torch.zeros(batch_size, self.decoder_lstm.hidden_size, device=device)
            else:
                h = hidden[0][-1]

            attended_visual, _ = self.attention(visual_features, h)
            decoder_input = torch.cat([word_embed, attended_visual], dim=1).unsqueeze(1)

            output, hidden = self.decoder_lstm(decoder_input, hidden)
            logits = self.output_proj(output.squeeze(1))

            next_token = logits.argmax(dim=1)
            generated.append(next_token)

            # Check if all sequences generated end token
            if (next_token == end_token).all():
                break

            input_tokens = torch.cat([input_tokens, next_token.unsqueeze(1)], dim=1)

        generated = torch.stack(generated, dim=1)
        return generated


def generate_vqa_data(n_samples=1000, img_size=64):
    """
    Generate synthetic VQA data.

    Images: patterns (vertical, horizontal, checkerboard)
    Questions: "What pattern?" or "What color dominant?"
    Answers: pattern type or color
    """
    print(f"Generating {n_samples} VQA samples...")

    np.random.seed(42)

    images = []
    questions = []
    answers = []

    # Vocabulary:
    # Questions: [PAD=0, "what"=1, "pattern"=2, "is"=3, "the"=4, "?"=5]
    # Answers: ["vertical"=0, "horizontal"=1, "checkerboard"=2]

    for _ in range(n_samples):
        pattern = np.random.randint(0, 3)

        # Create image
        img = np.zeros((3, img_size, img_size), dtype=np.float32)

        if pattern == 0:
            # Vertical stripes
            for i in range(0, img_size, 8):
                img[:, :, i:i+4] = 1.0
        elif pattern == 1:
            # Horizontal stripes
            for i in range(0, img_size, 8):
                img[:, i:i+4, :] = 1.0
        else:
            # Checkerboard
            for i in range(0, img_size, 8):
                for j in range(0, img_size, 8):
                    if (i + j) % 16 == 0:
                        img[:, i:i+8, j:j+8] = 1.0

        # Add noise
        img += np.random.randn(3, img_size, img_size) * 0.1
        img = np.clip(img, 0, 1)

        # Question: "what is the pattern?"
        question = [1, 3, 4, 2, 5, 0, 0, 0, 0, 0]  # Padded to length 10

        # Answer
        answer = pattern

        images.append(img)
        questions.append(question)
        answers.append(answer)

    return (np.array(images, dtype=np.float32),
            np.array(questions, dtype=np.int64),
            np.array(answers, dtype=np.int64))


def generate_captioning_data(n_samples=1000, img_size=64):
    """
    Generate synthetic image captioning data.

    Images: patterns
    Captions: "this is a [pattern] pattern"
    """
    print(f"Generating {n_samples} captioning samples...")

    np.random.seed(42)

    images = []
    captions = []

    # Vocabulary:
    # [PAD=0, <START>=1, <END>=2, "this"=3, "is"=4, "a"=5,
    #  "vertical"=6, "horizontal"=7, "checkerboard"=8, "pattern"=9]

    for _ in range(n_samples):
        pattern = np.random.randint(0, 3)

        # Create image
        img = np.zeros((3, img_size, img_size), dtype=np.float32)

        if pattern == 0:
            for i in range(0, img_size, 8):
                img[:, :, i:i+4] = 1.0
        elif pattern == 1:
            for i in range(0, img_size, 8):
                img[:, i:i+4, :] = 1.0
        else:
            for i in range(0, img_size, 8):
                for j in range(0, img_size, 8):
                    if (i + j) % 16 == 0:
                        img[:, i:i+8, j:j+8] = 1.0

        img += np.random.randn(3, img_size, img_size) * 0.1
        img = np.clip(img, 0, 1)

        # Caption: <START> this is a [pattern] pattern <END>
        pattern_token = 6 + pattern  # 6, 7, or 8
        caption = [1, 3, 4, 5, pattern_token, 9, 2, 0, 0, 0]  # Padded

        images.append(img)
        captions.append(caption)

    return np.array(images, dtype=np.float32), np.array(captions, dtype=np.int64)


class VQADataset(Dataset):
    """Dataset for VQA."""
    def __init__(self, images, questions, answers):
        self.images = torch.FloatTensor(images)
        self.questions = torch.LongTensor(questions)
        self.answers = torch.LongTensor(answers)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.questions[idx], self.answers[idx]


class CaptioningDataset(Dataset):
    """Dataset for captioning."""
    def __init__(self, images, captions):
        self.images = torch.FloatTensor(images)
        self.captions = torch.LongTensor(captions)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Input: captions without <END>, Target: captions without <START>
        return self.images[idx], self.captions[idx, :-1], self.captions[idx, 1:]


def train_vqa(model, train_loader, val_loader, epochs=40, lr=0.001):
    """Train VQA model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining VQA on {device}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    history = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}
    best_val_accuracy = 0

    for epoch in range(epochs):
        start_time = time.time()

        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for images, questions, answers in train_loader:
            images = images.to(device)
            questions = questions.to(device)
            answers = answers.to(device)

            logits, _ = model(images, questions)
            loss = criterion(logits, answers)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (logits.argmax(dim=1) == answers).sum().item()
            train_total += len(answers)

        train_loss /= len(train_loader)
        train_accuracy = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, questions, answers in val_loader:
                images = images.to(device)
                questions = questions.to(device)
                answers = answers.to(device)

                logits, _ = model(images, questions)
                loss = criterion(logits, answers)

                val_loss += loss.item()
                val_correct += (logits.argmax(dim=1) == answers).sum().item()
                val_total += len(answers)

        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_accuracy'].append(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_vqa.pth')

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] ({time.time()-start_time:.2f}s) - "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_accuracy:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_accuracy:.4f}")

    return history


def train_captioning(model, train_loader, val_loader, epochs=40, lr=0.001):
    """Train captioning model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining Captioning on {device}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')

    for epoch in range(epochs):
        start_time = time.time()

        # Training
        model.train()
        train_loss = 0

        for images, caption_input, caption_target in train_loader:
            images = images.to(device)
            caption_input = caption_input.to(device)
            caption_target = caption_target.to(device)

            logits = model(images, caption_input)
            loss = criterion(logits.view(-1, model.vocab_size), caption_target.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for images, caption_input, caption_target in val_loader:
                images = images.to(device)
                caption_input = caption_input.to(device)
                caption_target = caption_target.to(device)

                logits = model(images, caption_input)
                loss = criterion(logits.view(-1, model.vocab_size), caption_target.view(-1))

                val_loss += loss.item()

        val_loss /= len(val_loader)

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_captioning.pth')

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] ({time.time()-start_time:.2f}s) - "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    return history


def plot_training_curves(history, task_name):
    """Plot training curves."""
    if 'train_accuracy' in history:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        axes[0].plot(history['train_loss'], label='Train', linewidth=2)
        axes[0].plot(history['val_loss'], label='Validation', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'{task_name} - Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(history['train_accuracy'], label='Train', linewidth=2)
        axes[1].plot(history['val_accuracy'], label='Validation', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title(f'{task_name} - Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        plt.figure(figsize=(8, 4))
        plt.plot(history['train_loss'], label='Train', linewidth=2)
        plt.plot(history['val_loss'], label='Validation', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{task_name} - Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{task_name.lower().replace(" ", "_")}_curves.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main execution function."""
    print("="*70)
    print("Vision-Language Models")
    print("="*70)

    # ========== Visual Question Answering ==========
    print("\n" + "="*70)
    print("Task 1: Visual Question Answering (VQA)")
    print("="*70)

    # Generate VQA data
    print("\n1. Generating VQA data...")
    vqa_images, vqa_questions, vqa_answers = generate_vqa_data(n_samples=1500)

    n_train = int(0.7 * len(vqa_images))
    n_val = int(0.15 * len(vqa_images))

    vqa_train_dataset = VQADataset(
        vqa_images[:n_train], vqa_questions[:n_train], vqa_answers[:n_train]
    )
    vqa_val_dataset = VQADataset(
        vqa_images[n_train:n_train+n_val],
        vqa_questions[n_train:n_train+n_val],
        vqa_answers[n_train:n_train+n_val]
    )

    vqa_train_loader = DataLoader(vqa_train_dataset, batch_size=32, shuffle=True)
    vqa_val_loader = DataLoader(vqa_val_dataset, batch_size=32, shuffle=False)

    # Create VQA model
    print("\n2. Creating VQA model...")
    vqa_model = VQAModel(vocab_size=10, num_answers=3, embed_dim=256, hidden_dim=512)
    print(f"VQA parameters: {sum(p.numel() for p in vqa_model.parameters()):,}")

    # Train VQA
    print("\n3. Training VQA model...")
    vqa_history = train_vqa(vqa_model, vqa_train_loader, vqa_val_loader, epochs=40)

    print("\n4. Plotting VQA curves...")
    plot_training_curves(vqa_history, "VQA")

    # ========== Image Captioning ==========
    print("\n" + "="*70)
    print("Task 2: Image Captioning")
    print("="*70)

    # Generate captioning data
    print("\n1. Generating captioning data...")
    cap_images, cap_captions = generate_captioning_data(n_samples=1500)

    cap_train_dataset = CaptioningDataset(cap_images[:n_train], cap_captions[:n_train])
    cap_val_dataset = CaptioningDataset(
        cap_images[n_train:n_train+n_val], cap_captions[n_train:n_train+n_val]
    )

    cap_train_loader = DataLoader(cap_train_dataset, batch_size=32, shuffle=True)
    cap_val_loader = DataLoader(cap_val_dataset, batch_size=32, shuffle=False)

    # Create captioning model
    print("\n2. Creating captioning model...")
    cap_model = ImageCaptioningModel(vocab_size=10, embed_dim=256, hidden_dim=512)
    print(f"Captioning parameters: {sum(p.numel() for p in cap_model.parameters()):,}")

    # Train captioning
    print("\n3. Training captioning model...")
    cap_history = train_captioning(cap_model, cap_train_loader, cap_val_loader, epochs=40)

    print("\n4. Plotting captioning curves...")
    plot_training_curves(cap_history, "Captioning")

    # Test caption generation
    print("\n5. Testing caption generation...")
    cap_model.load_state_dict(torch.load('best_captioning.pth'))
    device = next(cap_model.parameters()).device

    test_images = torch.FloatTensor(cap_images[n_train+n_val:n_train+n_val+5]).to(device)
    generated_captions = cap_model.generate(test_images, start_token=1, end_token=2)

    print("\nGenerated captions (first 5 test images):")
    for i, caption in enumerate(generated_captions):
        print(f"Image {i}: {caption.cpu().numpy()}")

    print("\n" + "="*70)
    print("Vision-Language Models Complete!")
    print("="*70)
    print("\nKey Features:")
    print("✓ Visual Question Answering with cross-modal attention")
    print("✓ Image captioning with attention over spatial features")
    print("✓ Joint vision-language understanding")
    print("✓ Attention mechanism for interpretability")
    print("\nApplications: VQA, image captioning, visual reasoning, accessibility")


if __name__ == "__main__":
    main()
