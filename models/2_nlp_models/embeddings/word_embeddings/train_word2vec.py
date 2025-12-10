"""
Word2Vec Word Embeddings with Gensim

Word2Vec learns dense vector representations of words:
- CBOW (Continuous Bag of Words): predicts target word from context
- Skip-gram: predicts context words from target word
- Captures semantic relationships (king - man + woman ≈ queen)
- Foundation for many NLP tasks

This implementation includes:
- CBOW and Skip-gram training
- Word similarity and analogy tasks
- Visualization with t-SNE
- Hyperparameter comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import time
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import re
from collections import Counter


class TrainingCallback(CallbackAny2Vec):
    """Callback to track training progress."""
    def __init__(self):
        self.epoch = 0
        self.losses = []

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        self.losses.append(loss)
        if self.epoch % 5 == 0:
            print(f'Epoch {self.epoch}: loss = {loss:.4f}')
        self.epoch += 1


def generate_sample_corpus(n_documents=1000, vocab_size=500):
    """Generate synthetic text corpus for demonstration."""
    print(f"Generating {n_documents} documents with vocab size {vocab_size}...")

    np.random.seed(42)

    # Define some topic-specific words
    topics = {
        'technology': ['computer', 'software', 'algorithm', 'data', 'network', 'programming',
                      'artificial', 'intelligence', 'machine', 'learning', 'neural', 'system'],
        'nature': ['tree', 'forest', 'mountain', 'river', 'ocean', 'animal', 'bird',
                  'flower', 'plant', 'weather', 'climate', 'environment'],
        'food': ['cook', 'recipe', 'ingredient', 'restaurant', 'meal', 'breakfast',
                'dinner', 'lunch', 'delicious', 'flavor', 'taste', 'kitchen'],
        'sports': ['game', 'play', 'team', 'player', 'score', 'win', 'competition',
                  'tournament', 'champion', 'athlete', 'training', 'exercise'],
        'science': ['experiment', 'research', 'theory', 'hypothesis', 'observation',
                   'analysis', 'discovery', 'scientist', 'laboratory', 'study', 'method']
    }

    # Generate documents
    documents = []
    for i in range(n_documents):
        # Choose a topic
        topic = list(topics.keys())[i % len(topics)]
        topic_words = topics[topic]

        # Generate sentence with 10-20 words
        doc_length = np.random.randint(10, 21)
        doc = []

        # 70% topic-specific words, 30% random words
        for _ in range(doc_length):
            if np.random.rand() < 0.7:
                doc.append(np.random.choice(topic_words))
            else:
                # Add some common words
                common = ['the', 'a', 'an', 'is', 'are', 'was', 'were', 'been',
                         'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                         'can', 'could', 'should', 'may', 'might', 'must']
                doc.append(np.random.choice(common))

        documents.append(doc)

    return documents


def load_real_text_corpus(text):
    """Process real text into sentences."""
    # Simple preprocessing
    text = text.lower()
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)

    # Tokenize
    corpus = []
    for sentence in sentences:
        words = re.findall(r'\b\w+\b', sentence)
        if len(words) >= 3:  # Keep sentences with at least 3 words
            corpus.append(words)

    return corpus


def train_word2vec_cbow(corpus, vector_size=100, window=5, min_count=2, epochs=20):
    """Train Word2Vec with CBOW architecture."""
    print(f"\nTraining Word2Vec (CBOW)")
    print(f"Vector size: {vector_size}, Window: {window}, Min count: {min_count}")

    callback = TrainingCallback()

    model = Word2Vec(
        sentences=corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=0,  # 0 = CBOW, 1 = Skip-gram
        workers=4,
        epochs=epochs,
        callbacks=[callback]
    )

    return model


def train_word2vec_skipgram(corpus, vector_size=100, window=5, min_count=2, epochs=20):
    """Train Word2Vec with Skip-gram architecture."""
    print(f"\nTraining Word2Vec (Skip-gram)")
    print(f"Vector size: {vector_size}, Window: {window}, Min count: {min_count}")

    callback = TrainingCallback()

    model = Word2Vec(
        sentences=corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=1,  # Skip-gram
        workers=4,
        epochs=epochs,
        callbacks=[callback]
    )

    return model


def evaluate_word_similarity(model, word_pairs):
    """Evaluate word similarity."""
    print("\nWord Similarity Evaluation:")
    print("="*60)

    for word1, word2 in word_pairs:
        try:
            similarity = model.wv.similarity(word1, word2)
            print(f"{word1:15} <-> {word2:15} : {similarity:.4f}")
        except KeyError as e:
            print(f"{word1:15} <-> {word2:15} : Not in vocabulary")


def find_similar_words(model, word, topn=5):
    """Find most similar words."""
    try:
        similar = model.wv.most_similar(word, topn=topn)
        print(f"\nMost similar to '{word}':")
        for similar_word, score in similar:
            print(f"  {similar_word:20} : {score:.4f}")
    except KeyError:
        print(f"'{word}' not in vocabulary")


def word_analogy(model, positive, negative, topn=3):
    """Perform word analogy: positive - negative."""
    print(f"\nAnalogy: {' + '.join(positive)} - {' + '.join(negative)}")
    try:
        result = model.wv.most_similar(positive=positive, negative=negative, topn=topn)
        for word, score in result:
            print(f"  {word:20} : {score:.4f}")
    except KeyError as e:
        print(f"  Error: {e}")


def visualize_embeddings(model, words=None, n_words=50):
    """Visualize word embeddings using t-SNE."""
    if words is None:
        # Get most common words
        words = list(model.wv.index_to_key[:n_words])

    # Get vectors
    vectors = np.array([model.wv[word] for word in words])

    # Reduce to 2D using t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(words)-1))
    vectors_2d = tsne.fit_transform(vectors)

    # Plot
    plt.figure(figsize=(14, 10))

    # Color by first letter for variety
    colors = plt.cm.Set3(np.linspace(0, 1, 26))

    for i, word in enumerate(words):
        x, y = vectors_2d[i]
        color_idx = ord(word[0].lower()) - ord('a')
        if 0 <= color_idx < 26:
            color = colors[color_idx]
        else:
            color = 'gray'

        plt.scatter(x, y, c=[color], s=100, alpha=0.6)
        plt.annotate(word, (x, y), xytext=(5, 5), textcoords='offset points',
                    fontsize=9, alpha=0.8)

    plt.title('Word Embeddings Visualization (t-SNE)', fontsize=14)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('word2vec_embeddings.png', dpi=300, bbox_inches='tight')
    plt.show()


def compare_cbow_skipgram(corpus, vector_size=100, window=5, epochs=20):
    """Compare CBOW and Skip-gram architectures."""
    print("\n" + "="*70)
    print("Comparing CBOW vs Skip-gram")
    print("="*70)

    # Train both models
    start_time = time.time()
    cbow_model = train_word2vec_cbow(corpus, vector_size, window, min_count=2, epochs=epochs)
    cbow_time = time.time() - start_time

    start_time = time.time()
    sg_model = train_word2vec_skipgram(corpus, vector_size, window, min_count=2, epochs=epochs)
    sg_time = time.time() - start_time

    print("\n" + "="*70)
    print("Comparison Results")
    print("="*70)
    print(f"CBOW Training Time: {cbow_time:.2f}s")
    print(f"Skip-gram Training Time: {sg_time:.2f}s")
    print(f"CBOW Vocabulary Size: {len(cbow_model.wv)}")
    print(f"Skip-gram Vocabulary Size: {len(sg_model.wv)}")

    # Test similarity on common words
    test_words = [w for w in ['computer', 'algorithm', 'tree', 'forest', 'cook', 'recipe']
                  if w in cbow_model.wv and w in sg_model.wv]

    if len(test_words) >= 2:
        print(f"\nSimilarity between '{test_words[0]}' and '{test_words[1]}':")
        try:
            cbow_sim = cbow_model.wv.similarity(test_words[0], test_words[1])
            sg_sim = sg_model.wv.similarity(test_words[0], test_words[1])
            print(f"  CBOW: {cbow_sim:.4f}")
            print(f"  Skip-gram: {sg_sim:.4f}")
        except:
            print("  Unable to compute similarity")

    return cbow_model, sg_model


def hyperparameter_comparison(corpus):
    """Compare different hyperparameters."""
    print("\n" + "="*70)
    print("Hyperparameter Comparison")
    print("="*70)

    # Different vector sizes
    vector_sizes = [50, 100, 200]
    windows = [3, 5, 10]

    results = []

    for vec_size in vector_sizes:
        for window in windows:
            print(f"\nVector size: {vec_size}, Window: {window}")
            start_time = time.time()

            model = Word2Vec(
                sentences=corpus,
                vector_size=vec_size,
                window=window,
                min_count=2,
                sg=1,  # Skip-gram
                workers=4,
                epochs=10
            )

            train_time = time.time() - start_time
            vocab_size = len(model.wv)

            results.append({
                'vec_size': vec_size,
                'window': window,
                'time': train_time,
                'vocab': vocab_size
            })

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Training time
    data = np.array([[r['time'] for r in results if r['vec_size'] == vs]
                     for vs in vector_sizes])
    im1 = axes[0].imshow(data, cmap='YlOrRd', aspect='auto')
    axes[0].set_xticks(range(len(windows)))
    axes[0].set_xticklabels(windows)
    axes[0].set_yticks(range(len(vector_sizes)))
    axes[0].set_yticklabels(vector_sizes)
    axes[0].set_xlabel('Window Size')
    axes[0].set_ylabel('Vector Size')
    axes[0].set_title('Training Time (seconds)')
    plt.colorbar(im1, ax=axes[0])

    # Add text annotations
    for i in range(len(vector_sizes)):
        for j in range(len(windows)):
            text = axes[0].text(j, i, f'{data[i, j]:.1f}',
                               ha="center", va="center", color="black", fontsize=10)

    # Vocabulary size
    data = np.array([[r['vocab'] for r in results if r['vec_size'] == vs]
                     for vs in vector_sizes])
    im2 = axes[1].imshow(data, cmap='Blues', aspect='auto')
    axes[1].set_xticks(range(len(windows)))
    axes[1].set_xticklabels(windows)
    axes[1].set_yticks(range(len(vector_sizes)))
    axes[1].set_yticklabels(vector_sizes)
    axes[1].set_xlabel('Window Size')
    axes[1].set_ylabel('Vector Size')
    axes[1].set_title('Vocabulary Size')
    plt.colorbar(im2, ax=axes[1])

    for i in range(len(vector_sizes)):
        for j in range(len(windows)):
            text = axes[1].text(j, i, f'{data[i, j]}',
                               ha="center", va="center", color="black", fontsize=10)

    plt.tight_layout()
    plt.savefig('word2vec_hyperparameter_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main execution function."""
    print("="*70)
    print("Word2Vec Word Embeddings")
    print("="*70)

    # Generate corpus
    print("\n1. Generating corpus...")
    corpus = generate_sample_corpus(n_documents=2000, vocab_size=500)
    total_words = sum(len(doc) for doc in corpus)
    unique_words = len(set(word for doc in corpus for word in doc))
    print(f"Total documents: {len(corpus)}")
    print(f"Total words: {total_words}")
    print(f"Unique words: {unique_words}")

    # Compare CBOW and Skip-gram
    print("\n2. Comparing CBOW and Skip-gram...")
    cbow_model, sg_model = compare_cbow_skipgram(corpus, vector_size=100, window=5, epochs=30)

    # Evaluate word similarity
    print("\n3. Evaluating word similarity...")
    word_pairs = [
        ('computer', 'software'),
        ('algorithm', 'data'),
        ('tree', 'forest'),
        ('mountain', 'river'),
        ('cook', 'recipe'),
        ('game', 'play')
    ]
    evaluate_word_similarity(sg_model, word_pairs)

    # Find similar words
    print("\n4. Finding similar words...")
    test_words = ['computer', 'tree', 'cook', 'game']
    for word in test_words:
        if word in sg_model.wv:
            find_similar_words(sg_model, word, topn=5)

    # Word analogies
    print("\n5. Testing word analogies...")
    if all(w in sg_model.wv for w in ['computer', 'software', 'tree']):
        word_analogy(sg_model, ['computer'], ['software'], topn=3)

    # Visualize embeddings
    print("\n6. Visualizing embeddings...")
    common_words = [w for w in sg_model.wv.index_to_key[:50]]
    visualize_embeddings(sg_model, words=common_words)

    # Hyperparameter comparison
    print("\n7. Comparing hyperparameters...")
    hyperparameter_comparison(corpus)

    # Save model
    sg_model.save('word2vec_model.bin')
    print("\nModel saved to 'word2vec_model.bin'")

    print("\n" + "="*70)
    print("Word2Vec Training Complete!")
    print("="*70)
    print("\nKey Features:")
    print("✓ CBOW: Fast, good for frequent words")
    print("✓ Skip-gram: Better for rare words, semantic relationships")
    print("✓ Captures semantic similarity (king - man + woman ≈ queen)")
    print("✓ Foundation for transfer learning in NLP")
    print("\nBest for: Text classification, NER, sentiment analysis, document similarity")


if __name__ == "__main__":
    main()
