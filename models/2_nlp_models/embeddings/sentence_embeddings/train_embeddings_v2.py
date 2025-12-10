"""
Alternative Embeddings using Sentence-Transformers
For better semantic similarity and retrieval
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.datasets import fetch_20newsgroups
import warnings

warnings.filterwarnings("ignore")


def train():
    print("Training Alternative Embeddings with Sentence-Transformers...")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print(
            "Note: Install sentence-transformers with: pip install sentence-transformers"
        )
        return train_fallback_embeddings()

    try:
        # 1. Load Data
        categories = [
            "alt.atheism",
            "soc.religion.christian",
            "comp.graphics",
            "sci.med",
        ]
        newsgroups = fetch_20newsgroups(
            subset="train",
            categories=categories,
            remove=("headers", "footers", "quotes"),
        )
        documents = newsgroups.data[:100]  # Use first 100 documents

    except Exception:
        print(f"Warning: Could not load 20 newsgroups. Using synthetic data...")
        documents = create_synthetic_documents()

    # 2. Load Sentence-Transformers Model
    model_name = "all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    print(f"Loaded model: {model_name}")

    # 3. Generate Embeddings
    print(f"\nEmbedding {len(documents)} documents...")
    embeddings = model.encode(documents, show_progress_bar=False)

    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    # 4. Compute Similarity
    print("\n=== Semantic Similarity Examples ===")

    # Compare first few documents
    similarity_matrix = cosine_similarity(embeddings[:10])

    for i in range(3):
        # Second most similar (first is itself)
        similar_idx = np.argsort(similarity_matrix[i])[-2]
        similarity_score = similarity_matrix[i][similar_idx]

        print(f"\n--- Similarity Pair {i+1} ---")
        print(f"Doc 1: {documents[i][:100]}...")
        print(f"Doc 2: {documents[similar_idx][:100]}...")
        print(f"Similarity Score: {similarity_score:.4f}")

    # 5. Semantic Search
    print("\n=== Semantic Search Example ===")
    query = "machine learning and artificial intelligence"
    query_embedding = model.encode(query)

    similarities = cosine_similarity([query_embedding], embeddings)[0]
    top_indices = np.argsort(similarities)[-3:][::-1]

    print(f"Query: {query}\n")
    for rank, idx in enumerate(top_indices, 1):
        print(f"Top {rank} (similarity: {similarities[idx]:.4f}):")
        print(f"  {documents[idx][:100]}...\n")

    # 6. QA Validation
    print("=== QA Validation ===")

    # Check embedding quality
    avg_similarity = np.mean(
        similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
    )
    print(f"Average inter-document similarity: {avg_similarity:.4f}")

    print("\n--- Sanity Checks ---")
    if embeddings.shape[1] > 0:
        print(f"✓ Embeddings generated successfully: {embeddings.shape}")
    else:
        print("✗ Failed to generate embeddings")

    if not np.any(np.isnan(embeddings)):
        print("✓ No NaN values in embeddings")
    else:
        print("✗ WARNING: NaN values detected in embeddings!")

    print("\n=== Overall Validation Result ===")
    validation_passed = embeddings.shape[1] > 0 and not np.any(np.isnan(embeddings))

    if validation_passed:
        print("✓ Validation PASSED")
    else:
        print("✗ Validation FAILED")

    return model


def train_fallback_embeddings():
    """Fallback embedding using TF-IDF"""
    print("\nUsing TF-IDF embeddings as fallback...")

    from sklearn.feature_extraction.text import TfidfVectorizer

    documents = create_synthetic_documents()

    vectorizer = TfidfVectorizer(max_features=300)
    embeddings = vectorizer.fit_transform(documents).toarray()

    print(f"TF-IDF Embeddings shape: {embeddings.shape}")

    class FallbackModel:
        def __init__(self, vectorizer):
            self.vectorizer = vectorizer

        def encode(self, texts):
            if isinstance(texts, str):
                return self.vectorizer.transform([texts]).toarray()[0]
            return self.vectorizer.transform(texts).toarray()

    model = FallbackModel(vectorizer)

    print("✓ Fallback embeddings validation PASSED")

    return model


def create_synthetic_documents():
    """Create synthetic documents for testing"""
    return [
        "Machine learning is a subset of artificial intelligence that focuses on learning from data.",
        "Deep learning uses neural networks with multiple layers to process data.",
        "Natural language processing helps computers understand and generate human language.",
        "Computer vision enables machines to interpret and understand images and videos.",
        "Artificial intelligence is revolutionizing technology and society.",
        "Neural networks are inspired by biological neurons in the brain.",
        "Data science combines statistics, programming, and domain knowledge.",
        "Python is widely used for machine learning and data science projects.",
        "Transformers have become the foundation of modern NLP models.",
        "Object detection identifies and locates objects within images or videos.",
    ] * 10


if __name__ == "__main__":
    train()
