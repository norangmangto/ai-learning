"""
Sentence Embeddings Training using Sentence-Transformers

This script demonstrates:
1. Fine-tuning sentence embeddings on custom data
2. Using pre-trained SBERT models
3. Evaluating sentence similarity
4. Creating a sentence embedding model from scratch

Dataset: SNLI (Stanford Natural Language Inference) or custom pairs
Models: SBERT, SimCSE-style contrastive learning
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    evaluation,
    models
)
from datasets import load_dataset
import numpy as np
from pathlib import Path
import time
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Configuration
CONFIG = {
    'base_model': 'distilbert-base-uncased',  # Base transformer
    'max_seq_length': 128,
    'embedding_dim': 768,
    'batch_size': 16,
    'epochs': 1,  # Use more for production
    'learning_rate': 2e-5,
    'warmup_steps': 100,
    'evaluation_steps': 500,
    'output_dir': 'models/sentence_bert',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'max_samples': 10000,  # Limit for demo
}

def load_training_data(max_samples=10000):
    """Load NLI dataset for sentence embedding training"""
    print("Loading SNLI dataset...")

    try:
        # Load SNLI (Stanford Natural Language Inference)
        dataset = load_dataset('snli', split='train', streaming=True)

        train_examples = []

        for i, item in enumerate(dataset):
            if i >= max_samples:
                break

            # Skip examples with no label
            if item['label'] == -1:
                continue

            # Create sentence pairs with labels
            # 0: entailment (similar), 1: neutral, 2: contradiction
            premise = item['premise']
            hypothesis = item['hypothesis']
            label = item['label']

            # Convert to similarity score (0-1)
            # entailment -> 1.0, neutral -> 0.5, contradiction -> 0.0
            score_map = {0: 1.0, 1: 0.5, 2: 0.0}
            score = score_map.get(label, 0.5)

            train_examples.append(InputExample(
                texts=[premise, hypothesis],
                label=score
            ))

            if (i + 1) % 1000 == 0:
                print(f"Loaded {len(train_examples)} examples...")

        print(f"Loaded {len(train_examples)} training examples")
        return train_examples

    except Exception as e:
        print(f"Error loading SNLI: {e}")
        print("Using sample data instead...")

        # Fallback to sample data
        sample_examples = [
            InputExample(texts=['The cat sits on the mat', 'A cat is sitting on a rug'], label=0.9),
            InputExample(texts=['A man is playing guitar', 'A person is making music'], label=0.8),
            InputExample(texts=['The weather is nice', 'It is raining heavily'], label=0.2),
            InputExample(texts=['She loves reading books', 'She enjoys literature'], label=0.85),
            InputExample(texts=['The car is fast', 'The vehicle has high speed'], label=0.9),
            InputExample(texts=['I am learning Python', 'I study programming'], label=0.7),
            InputExample(texts=['The movie was boring', 'The film was exciting'], label=0.1),
            InputExample(texts=['He runs every morning', 'He exercises daily'], label=0.75),
            InputExample(texts=['The food tastes good', 'The meal is delicious'], label=0.85),
            InputExample(texts=['She is a doctor', 'She works at a restaurant'], label=0.2),
        ]

        # Duplicate for more training data
        train_examples = sample_examples * 100

        return train_examples

def create_sentence_transformer():
    """Create sentence transformer model from scratch"""
    print("\n" + "="*80)
    print("CREATING SENTENCE TRANSFORMER MODEL")
    print("="*80)

    print(f"\nBase model: {CONFIG['base_model']}")

    # Define model architecture
    word_embedding_model = models.Transformer(
        CONFIG['base_model'],
        max_seq_length=CONFIG['max_seq_length']
    )

    # Apply mean pooling to get sentence embeddings
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False
    )

    # Create the sentence transformer
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    print(f"Model created with embedding dimension: {model.get_sentence_embedding_dimension()}")

    return model

def train_sentence_embeddings(model, train_examples):
    """Fine-tune sentence embeddings"""
    print("\n" + "="*80)
    print("TRAINING SENTENCE EMBEDDINGS")
    print("="*80)

    print(f"\nConfiguration:")
    print(f"  Training examples: {len(train_examples)}")
    print(f"  Batch size: {CONFIG['batch_size']}")
    print(f"  Epochs: {CONFIG['epochs']}")
    print(f"  Learning rate: {CONFIG['learning_rate']}")
    print(f"  Device: {CONFIG['device']}")

    # Create data loader
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=CONFIG['batch_size']
    )

    # Define loss function
    # CosineSimilarityLoss for regression on similarity scores
    train_loss = losses.CosineSimilarityLoss(model)

    print("\nTraining loss: CosineSimilarityLoss")

    # Train
    start_time = time.time()

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=CONFIG['epochs'],
        warmup_steps=CONFIG['warmup_steps'],
        output_path=CONFIG['output_dir'],
        show_progress_bar=True,
    )

    training_time = time.time() - start_time

    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Model saved to: {CONFIG['output_dir']}")

    return model

def evaluate_embeddings(model):
    """Evaluate sentence embeddings"""
    print("\n" + "="*80)
    print("EVALUATING SENTENCE EMBEDDINGS")
    print("="*80)

    # Test sentences
    test_sentences = [
        # Similar pairs
        "The cat is sitting on the mat",
        "A cat sits on a rug",

        # Different meaning
        "The weather is nice today",
        "I love programming in Python",

        # Paraphrases
        "She enjoys reading books",
        "She loves to read literature",

        # Technical vs casual
        "Neural networks learn from data",
        "AI systems improve with experience",

        # Unrelated
        "The car is red",
        "Mathematics is difficult"
    ]

    # Encode sentences
    print("\nEncoding test sentences...")
    embeddings = model.encode(test_sentences, convert_to_tensor=True)

    print(f"Embedding shape: {embeddings.shape}")

    # Compute similarity matrix
    print("\nComputing similarity matrix...")
    similarities = cosine_similarity(embeddings.cpu().numpy())

    # Display results
    print("\n" + "="*80)
    print("SIMILARITY MATRIX")
    print("="*80)

    for i, sent1 in enumerate(test_sentences):
        print(f"\nS{i+1}: {sent1[:50]}...")
        for j, sent2 in enumerate(test_sentences):
            if i < j:  # Only show upper triangle
                sim = similarities[i][j]
                print(f"  S{i+1} <-> S{j+1}: {sim:.4f}")

    # Highlight interesting pairs
    print("\n" + "="*80)
    print("KEY OBSERVATIONS")
    print("="*80)

    print("\nHigh similarity (paraphrases):")
    print(f"  S1 <-> S2: {similarities[0][1]:.4f} (cat on mat)")
    print(f"  S5 <-> S6: {similarities[4][5]:.4f} (reading books)")

    print("\nLow similarity (unrelated):")
    print(f"  S1 <-> S4: {similarities[0][3]:.4f} (cat vs weather)")
    print(f"  S9 <-> S10: {similarities[8][9]:.4f} (car vs math)")

    # Visualize similarity matrix
    visualize_similarity_matrix(similarities, test_sentences)

    return embeddings, similarities

def visualize_similarity_matrix(similarities, sentences):
    """Visualize sentence similarity matrix"""
    plt.figure(figsize=(12, 10))

    # Create heatmap
    plt.imshow(similarities, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    plt.colorbar(label='Cosine Similarity')

    # Add labels
    labels = [f"S{i+1}" for i in range(len(sentences))]
    plt.xticks(range(len(sentences)), labels, rotation=0)
    plt.yticks(range(len(sentences)), labels)

    # Add text annotations
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            text = plt.text(j, i, f'{similarities[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)

    plt.title('Sentence Similarity Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Sentence Index')
    plt.ylabel('Sentence Index')
    plt.tight_layout()

    # Save
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / 'sentence_similarity_matrix.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSimilarity matrix saved to: {output_file}")

    plt.close()

def demonstrate_use_cases(model):
    """Demonstrate practical use cases"""
    print("\n" + "="*80)
    print("PRACTICAL USE CASES")
    print("="*80)

    # Use Case 1: Semantic Search
    print("\n1. SEMANTIC SEARCH")
    print("-" * 80)

    documents = [
        "Python is a popular programming language for data science",
        "Machine learning models learn patterns from data",
        "Neural networks are inspired by biological neurons",
        "The weather forecast predicts rain tomorrow",
        "Natural language processing helps computers understand text"
    ]

    query = "How do computers learn from data?"

    print(f"Query: {query}")
    print(f"\nSearching {len(documents)} documents...")

    # Encode
    query_embedding = model.encode(query, convert_to_tensor=True)
    doc_embeddings = model.encode(documents, convert_to_tensor=True)

    # Compute similarities
    similarities = torch.nn.functional.cosine_similarity(
        query_embedding.unsqueeze(0), doc_embeddings
    )

    # Get top results
    top_k = 3
    top_indices = torch.argsort(similarities, descending=True)[:top_k]

    print(f"\nTop {top_k} results:")
    for i, idx in enumerate(top_indices, 1):
        print(f"{i}. Score: {similarities[idx]:.4f}")
        print(f"   {documents[idx]}")

    # Use Case 2: Duplicate Detection
    print("\n2. DUPLICATE DETECTION")
    print("-" * 80)

    sentences = [
        "How to learn Python programming?",
        "What is the best way to study Python?",
        "Python programming tutorial",
        "Machine learning basics",
        "Introduction to machine learning"
    ]

    print("Finding duplicates (threshold: 0.7)...")

    embeddings = model.encode(sentences, convert_to_tensor=True)
    sim_matrix = torch.nn.functional.cosine_similarity(
        embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2
    )

    threshold = 0.7
    duplicates = []

    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            if sim_matrix[i][j] > threshold:
                duplicates.append((i, j, sim_matrix[i][j].item()))

    if duplicates:
        print(f"\nFound {len(duplicates)} potential duplicates:")
        for i, j, score in duplicates:
            print(f"\nSimilarity: {score:.4f}")
            print(f"  A: {sentences[i]}")
            print(f"  B: {sentences[j]}")
    else:
        print("No duplicates found")

    # Use Case 3: Clustering
    print("\n3. SENTENCE CLUSTERING")
    print("-" * 80)

    texts = [
        "Python programming tutorial",
        "Learn Python basics",
        "Machine learning course",
        "Deep learning fundamentals",
        "Cooking recipes for beginners",
        "How to bake a cake",
        "Travel guide to Europe",
        "Best places to visit in Asia"
    ]

    print(f"Clustering {len(texts)} sentences...")

    from sklearn.cluster import KMeans

    embeddings = model.encode(texts)
    n_clusters = 3

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    print(f"\nClusters (k={n_clusters}):")
    for cluster_id in range(n_clusters):
        print(f"\nCluster {cluster_id + 1}:")
        cluster_texts = [texts[i] for i in range(len(texts)) if labels[i] == cluster_id]
        for text in cluster_texts:
            print(f"  - {text}")

def main():
    print("="*80)
    print("SENTENCE EMBEDDINGS WITH SENTENCE-TRANSFORMERS")
    print("="*80)

    print(f"\nDevice: {CONFIG['device']}")

    # Option 1: Use pre-trained model
    print("\n" + "="*80)
    print("OPTION 1: PRE-TRAINED MODEL (No Training)")
    print("="*80)

    pretrained_model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"Loaded pre-trained model: all-MiniLM-L6-v2")
    print(f"Embedding dimension: {pretrained_model.get_sentence_embedding_dimension()}")

    # Evaluate pre-trained model
    evaluate_embeddings(pretrained_model)

    # Demonstrate use cases
    demonstrate_use_cases(pretrained_model)

    # Option 2: Fine-tune on custom data
    print("\n" + "="*80)
    print("OPTION 2: FINE-TUNED MODEL (With Training)")
    print("="*80)

    print("\nNote: Fine-tuning requires more time and data.")
    print("For production, use larger datasets and more epochs.")

    response = input("\nDo you want to fine-tune a model? (y/n): ").strip().lower()

    if response == 'y':
        # Load training data
        train_examples = load_training_data(CONFIG['max_samples'])

        # Create model
        model = create_sentence_transformer()

        # Train
        model = train_sentence_embeddings(model, train_examples)

        # Evaluate
        evaluate_embeddings(model)

    print("\n" + "="*80)
    print("COMPLETED")
    print("="*80)

    print("\nSentence Embeddings Summary:")
    print("✓ Convert sentences to fixed-size vectors")
    print("✓ Capture semantic meaning beyond keywords")
    print("✓ Pre-trained models available (no training needed)")
    print("✓ Can fine-tune on domain-specific data")
    print("✓ Fast inference (<100ms per sentence)")

    print("\nKey Applications:")
    print("- Semantic search and information retrieval")
    print("- Duplicate detection and clustering")
    print("- Question answering systems")
    print("- Text similarity and matching")
    print("- Recommendation systems")
    print("- Zero-shot classification")

    print("\nPopular Models:")
    print("- all-MiniLM-L6-v2: Fast, good quality")
    print("- all-mpnet-base-v2: Best quality")
    print("- paraphrase-multilingual: 50+ languages")
    print("- msmarco-distilbert: Information retrieval")

if __name__ == '__main__':
    main()
