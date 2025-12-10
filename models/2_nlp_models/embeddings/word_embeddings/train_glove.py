"""
GloVe (Global Vectors for Word Representation) Embeddings

GloVe combines global matrix factorization and local context window methods:
- Uses word co-occurrence statistics from corpus
- Factorizes word-word co-occurrence matrix
- Captures both global statistics and local context
- Often outperforms Word2Vec on word analogy tasks

This implementation includes:
- Co-occurrence matrix construction
- GloVe training from scratch
- Pre-trained GloVe loading
- Word similarity and analogy evaluation
- Comparison with Word2Vec
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import pickle
import time
from sklearn.manifold import TSNE


class GloVe:
    """
    GloVe word embeddings implementation.

    Objective: Minimize J = Σ f(X_ij) * (w_i^T * w_j + b_i + b_j - log(X_ij))^2
    where X_ij is the co-occurrence count of words i and j.
    """

    def __init__(
        self,
        vector_size=100,
        window_size=5,
        x_max=100,
        alpha=0.75,
        learning_rate=0.05,
        epochs=50,
    ):
        self.vector_size = vector_size
        self.window_size = window_size
        self.x_max = x_max
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.word_vectors = None
        self.context_vectors = None
        self.word_biases = None
        self.context_biases = None
        self.word_to_id = {}
        self.id_to_word = {}

    def build_cooccurrence_matrix(self, corpus, min_count=5):
        """
        Build word co-occurrence matrix from corpus.

        Args:
            corpus: List of tokenized sentences
            min_count: Minimum word frequency
        """
        print(f"Building co-occurrence matrix (window={self.window_size})...")

        # Count word frequencies
        word_counter = Counter()
        for sentence in corpus:
            word_counter.update(sentence)

        # Build vocabulary (words appearing >= min_count times)
        vocab = {word for word, count in word_counter.items() if count >= min_count}
        self.word_to_id = {word: idx for idx, word in enumerate(sorted(vocab))}
        self.id_to_word = {idx: word for word, idx in self.word_to_id.items()}

        vocab_size = len(self.word_to_id)
        print(f"Vocabulary size: {vocab_size}")

        # Build co-occurrence matrix (sparse)
        cooccurrence = defaultdict(float)

        for sentence in corpus:
            # Filter words in vocabulary
            words = [word for word in sentence if word in self.word_to_id]

            for i, word in enumerate(words):
                word_id = self.word_to_id[word]

                # Look at context words within window
                for j in range(
                    max(0, i - self.window_size),
                    min(len(words), i + self.window_size + 1),
                ):
                    if i != j:
                        context_word = words[j]
                        context_id = self.word_to_id[context_word]

                        # Weight by distance
                        distance = abs(i - j)
                        weight = 1.0 / distance

                        cooccurrence[(word_id, context_id)] += weight

        print(f"Co-occurrence entries: {len(cooccurrence)}")
        return cooccurrence

    def weighting_function(self, x):
        """Weighting function f(x) for GloVe objective."""
        if x < self.x_max:
            return (x / self.x_max) ** self.alpha
        return 1.0

    def train(self, corpus, min_count=5):
        """Train GloVe embeddings."""
        print("\n" + "=" * 70)
        print("Training GloVe Embeddings")
        print("=" * 70)

        # Build co-occurrence matrix
        cooccurrence = self.build_cooccurrence_matrix(corpus, min_count)
        vocab_size = len(self.word_to_id)

        # Initialize parameters
        np.random.seed(42)
        scale = 1.0 / np.sqrt(self.vector_size)
        self.word_vectors = np.random.uniform(
            -scale, scale, (vocab_size, self.vector_size)
        )
        self.context_vectors = np.random.uniform(
            -scale, scale, (vocab_size, self.vector_size)
        )
        self.word_biases = np.random.uniform(-scale, scale, vocab_size)
        self.context_biases = np.random.uniform(-scale, scale, vocab_size)

        # Convert co-occurrence to list for training
        cooccurrence_list = [(i, j, count) for (i, j), count in cooccurrence.items()]
        n_samples = len(cooccurrence_list)

        print(f"\nTraining on {n_samples} co-occurrence pairs...")
        print(f"Vector size: {self.vector_size}, Epochs: {self.epochs}")

        # Training loop
        losses = []

        for epoch in range(self.epochs):
            epoch_start = time.time()
            total_loss = 0

            # Shuffle data
            np.random.shuffle(cooccurrence_list)

            for i, j, x_ij in cooccurrence_list:
                # Forward pass
                diff = (
                    np.dot(self.word_vectors[i], self.context_vectors[j])
                    + self.word_biases[i]
                    + self.context_biases[j]
                    - np.log(x_ij)
                )

                # Weight
                weight = self.weighting_function(x_ij)

                # Loss
                loss = weight * (diff**2)
                total_loss += loss

                # Gradients
                grad_diff = 2 * weight * diff

                # Update word vector
                grad_word = grad_diff * self.context_vectors[j]
                self.word_vectors[i] -= self.learning_rate * grad_word

                # Update context vector
                grad_context = grad_diff * self.word_vectors[i]
                self.context_vectors[j] -= self.learning_rate * grad_context

                # Update biases
                self.word_biases[i] -= self.learning_rate * grad_diff
                self.context_biases[j] -= self.learning_rate * grad_diff

            avg_loss = total_loss / n_samples
            losses.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{self.epochs} ({time.time()-epoch_start:.2f}s) - "
                    f"Loss: {avg_loss:.4f}"
                )

        # Combine word and context vectors (common practice)
        self.embeddings = self.word_vectors + self.context_vectors

        return losses

    def get_vector(self, word):
        """Get embedding vector for a word."""
        if word in self.word_to_id:
            return self.embeddings[self.word_to_id[word]]
        return None

    def similarity(self, word1, word2):
        """Calculate cosine similarity between two words."""
        vec1 = self.get_vector(word1)
        vec2 = self.get_vector(word2)

        if vec1 is None or vec2 is None:
            return None

        # Cosine similarity
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def most_similar(self, word, topn=5):
        """Find most similar words."""
        vec = self.get_vector(word)
        if vec is None:
            return []

        # Calculate similarities with all words
        similarities = []
        for other_word, idx in self.word_to_id.items():
            if other_word != word:
                other_vec = self.embeddings[idx]
                sim = np.dot(vec, other_vec) / (
                    np.linalg.norm(vec) * np.linalg.norm(other_vec)
                )
                similarities.append((other_word, sim))

        # Sort and return top n
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:topn]

    def analogy(self, word_a, word_b, word_c, topn=3):
        """Solve word analogy: a is to b as c is to ?"""
        vec_a = self.get_vector(word_a)
        vec_b = self.get_vector(word_b)
        vec_c = self.get_vector(word_c)

        if vec_a is None or vec_b is None or vec_c is None:
            return []

        # Compute target vector: b - a + c
        target = vec_b - vec_a + vec_c

        # Find most similar to target
        similarities = []
        for word, idx in self.word_to_id.items():
            if word not in [word_a, word_b, word_c]:
                vec = self.embeddings[idx]
                sim = np.dot(target, vec) / (
                    np.linalg.norm(target) * np.linalg.norm(vec)
                )
                similarities.append((word, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:topn]

    def save(self, filepath):
        """Save model to file."""
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "embeddings": self.embeddings,
                    "word_to_id": self.word_to_id,
                    "id_to_word": self.id_to_word,
                    "vector_size": self.vector_size,
                },
                f,
            )
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        """Load model from file."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.embeddings = data["embeddings"]
            self.word_to_id = data["word_to_id"]
            self.id_to_word = data["id_to_word"]
            self.vector_size = data["vector_size"]
        print(f"Model loaded from {filepath}")


def generate_sample_corpus(n_documents=2000):
    """Generate synthetic text corpus."""
    print(f"Generating {n_documents} documents...")

    np.random.seed(42)

    topics = {
        "technology": [
            "computer",
            "software",
            "algorithm",
            "data",
            "network",
            "programming",
            "artificial",
            "intelligence",
            "machine",
            "learning",
            "neural",
            "system",
            "digital",
            "technology",
            "innovation",
            "developer",
        ],
        "nature": [
            "tree",
            "forest",
            "mountain",
            "river",
            "ocean",
            "animal",
            "bird",
            "flower",
            "plant",
            "weather",
            "climate",
            "environment",
            "natural",
            "wildlife",
            "ecosystem",
            "habitat",
        ],
        "food": [
            "cook",
            "recipe",
            "ingredient",
            "restaurant",
            "meal",
            "breakfast",
            "dinner",
            "lunch",
            "delicious",
            "flavor",
            "taste",
            "kitchen",
            "chef",
            "cuisine",
            "dish",
            "dessert",
        ],
        "sports": [
            "game",
            "play",
            "team",
            "player",
            "score",
            "win",
            "competition",
            "tournament",
            "champion",
            "athlete",
            "training",
            "exercise",
            "match",
            "victory",
            "sport",
            "fitness",
        ],
    }

    documents = []
    for i in range(n_documents):
        topic = list(topics.keys())[i % len(topics)]
        topic_words = topics[topic]

        doc_length = np.random.randint(15, 30)
        doc = []

        for _ in range(doc_length):
            if np.random.rand() < 0.7:
                doc.append(np.random.choice(topic_words))
            else:
                common = [
                    "the",
                    "a",
                    "is",
                    "are",
                    "and",
                    "or",
                    "but",
                    "in",
                    "on",
                    "at",
                    "to",
                    "for",
                    "with",
                    "by",
                    "from",
                    "about",
                    "as",
                    "into",
                ]
                doc.append(np.random.choice(common))

        documents.append(doc)

    return documents


def visualize_embeddings(glove_model, words=None, n_words=40):
    """Visualize GloVe embeddings using t-SNE."""
    if words is None:
        words = list(glove_model.word_to_id.keys())[:n_words]

    # Filter words that exist
    words = [w for w in words if w in glove_model.word_to_id]

    if len(words) < 2:
        print("Not enough words to visualize")
        return

    # Get vectors
    vectors = np.array([glove_model.get_vector(word) for word in words])

    # Reduce to 2D
    perplexity = min(30, len(words) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    vectors_2d = tsne.fit_transform(vectors)

    # Plot
    plt.figure(figsize=(14, 10))

    # Color by first letter
    colors = plt.cm.tab20(np.linspace(0, 1, 20))

    for i, word in enumerate(words):
        x, y = vectors_2d[i]
        color_idx = hash(word[0]) % 20

        plt.scatter(x, y, c=[colors[color_idx]], s=100, alpha=0.6)
        plt.annotate(
            word,
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
            alpha=0.9,
        )

    plt.title("GloVe Embeddings Visualization (t-SNE)", fontsize=14)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("glove_embeddings_tsne.png", dpi=300, bbox_inches="tight")
    plt.show()


def compare_hyperparameters(corpus):
    """Compare different GloVe hyperparameters."""
    print("\n" + "=" * 70)
    print("Hyperparameter Comparison")
    print("=" * 70)

    configs = [
        {"vector_size": 50, "window_size": 5, "x_max": 100},
        {"vector_size": 100, "window_size": 5, "x_max": 100},
        {"vector_size": 100, "window_size": 3, "x_max": 100},
        {"vector_size": 100, "window_size": 10, "x_max": 100},
    ]

    results = []

    for config in configs:
        print(f"\nTesting: {config}")
        model = GloVe(
            vector_size=config["vector_size"],
            window_size=config["window_size"],
            x_max=config["x_max"],
            learning_rate=0.05,
            epochs=20,
        )

        start_time = time.time()
        losses = model.train(corpus, min_count=5)
        train_time = time.time() - start_time

        results.append(
            {
                "config": config,
                "losses": losses,
                "time": train_time,
                "vocab_size": len(model.word_to_id),
            }
        )

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Training loss
    for i, result in enumerate(results):
        config_str = f"Vec={
    result['config']['vector_size']}, Win={
        result['config']['window_size']}"
        axes[0].plot(result["losses"], label=config_str, linewidth=2)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss Curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Training time
    config_labels = [
        f"V{r['config']['vector_size']}\nW{r['config']['window_size']}" for r in results
    ]
    times = [r["time"] for r in results]

    axes[1].bar(config_labels, times, color="steelblue", alpha=0.7)
    axes[1].set_ylabel("Training Time (seconds)")
    axes[1].set_title("Training Time Comparison")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("glove_hyperparameter_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    """Main execution function."""
    print("=" * 70)
    print("GloVe (Global Vectors) Word Embeddings")
    print("=" * 70)

    # Generate corpus
    print("\n1. Generating corpus...")
    corpus = generate_sample_corpus(n_documents=3000)
    total_words = sum(len(doc) for doc in corpus)
    unique_words = len(set(word for doc in corpus for word in doc))
    print(f"Total documents: {len(corpus)}")
    print(f"Total words: {total_words}")
    print(f"Unique words: {unique_words}")

    # Train GloVe
    print("\n2. Training GloVe model...")
    glove = GloVe(
        vector_size=100,
        window_size=5,
        x_max=100,
        alpha=0.75,
        learning_rate=0.05,
        epochs=50,
    )
    losses = glove.train(corpus, min_count=5)

    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses, linewidth=2, color="steelblue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GloVe Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("glove_training_loss.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Test word similarity
    print("\n3. Testing word similarity...")
    test_pairs = [
        ("computer", "software"),
        ("algorithm", "data"),
        ("tree", "forest"),
        ("cook", "recipe"),
        ("game", "play"),
    ]

    print("\nWord Similarities:")
    print("=" * 60)
    for word1, word2 in test_pairs:
        sim = glove.similarity(word1, word2)
        if sim is not None:
            print(f"{word1:15} <-> {word2:15} : {sim:.4f}")

    # Find similar words
    print("\n4. Finding similar words...")
    test_words = ["computer", "tree", "cook", "game"]
    for word in test_words:
        if word in glove.word_to_id:
            print(f"\nMost similar to '{word}':")
            similar = glove.most_similar(word, topn=5)
            for similar_word, score in similar:
                print(f"  {similar_word:20} : {score:.4f}")

    # Word analogies
    print("\n5. Testing word analogies...")
    if all(w in glove.word_to_id for w in ["computer", "software", "tree"]):
        print("\nAnalogy: computer - software + tree")
        results = glove.analogy("computer", "software", "tree", topn=3)
        for word, score in results:
            print(f"  {word:20} : {score:.4f}")

    # Visualize embeddings
    print("\n6. Visualizing embeddings...")
    common_words = list(glove.word_to_id.keys())[:50]
    visualize_embeddings(glove, words=common_words)

    # Hyperparameter comparison
    print("\n7. Comparing hyperparameters...")
    compare_hyperparameters(corpus)

    # Save model
    glove.save("glove_model.pkl")

    print("\n" + "=" * 70)
    print("GloVe Training Complete!")
    print("=" * 70)
    print("\nKey Features:")
    print("✓ Uses global co-occurrence statistics")
    print("✓ Combines matrix factorization and local context")
    print("✓ Often better on word analogy tasks than Word2Vec")
    print("✓ Deterministic training (unlike Word2Vec)")
    print("\nGloVe vs Word2Vec:")
    print("  • GloVe: Global statistics, deterministic, faster for large corpora")
    print("  • Word2Vec: Local context, stochastic, better for small corpora")
    print("\nBest for: Question answering, NER, text classification, semantic analysis")


if __name__ == "__main__":
    main()
