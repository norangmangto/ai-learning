"""
Word Embeddings Implementation using Gensim

This script demonstrates training and using word embeddings:
1. Train Word2Vec from scratch on text corpus
2. Load pre-trained GloVe embeddings
3. Find similar words and analogies
4. Visualize embeddings with dimensionality reduction

Dataset: Text8 corpus (Wikipedia dump)
Models: Word2Vec (CBOW & Skip-gram), GloVe
"""

from gensim.models import Word2Vec, KeyedVectors
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import urllib.request
import zipfile
import os
from pathlib import Path
import time

# Configuration
CONFIG = {
    'vector_size': 100,  # Embedding dimension
    'window': 5,  # Context window size
    'min_count': 5,  # Minimum word frequency
    'workers': 4,  # Parallel workers
    'epochs': 5,  # Training epochs
    'sg': 1,  # 0=CBOW, 1=Skip-gram
    'negative': 5,  # Negative sampling
    'sample': 1e-3,  # Downsampling frequent words
    'alpha': 0.025,  # Initial learning rate
    'min_alpha': 0.0001,  # Minimum learning rate
}

def download_text8():
    """Download text8 dataset"""
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)

    zip_file = data_dir / 'text8.zip'
    text_file = data_dir / 'text8'

    if text_file.exists():
        print(f"Text8 dataset already exists at {text_file}")
        return str(text_file)

    print("Downloading text8 dataset...")
    url = 'http://mattmahoney.net/dc/text8.zip'

    try:
        urllib.request.urlretrieve(url, zip_file)

        print("Extracting text8 dataset...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_dir)

        os.remove(zip_file)
        print(f"Dataset ready at {text_file}")

        return str(text_file)

    except Exception as e:
        print(f"Error downloading text8: {e}")
        return None

def load_corpus(file_path):
    """Load and preprocess corpus"""
    print(f"\nLoading corpus from {file_path}...")

    with open(file_path, 'r') as f:
        text = f.read()

    # Split into sentences (simple approach)
    # Text8 is already tokenized with spaces
    words = text.split()

    # Create sentences (chunks of words)
    sentence_length = 10
    sentences = []

    for i in range(0, len(words), sentence_length):
        sentence = words[i:i+sentence_length]
        if len(sentence) > 0:
            sentences.append(sentence)

    print(f"Loaded {len(sentences)} sentences")
    print(f"Total words: {len(words)}")
    print(f"Sample sentence: {' '.join(sentences[0])}")

    return sentences

def train_word2vec(sentences, config):
    """Train Word2Vec model"""
    print("\n" + "="*80)
    print("TRAINING WORD2VEC MODEL")
    print("="*80)
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    print(f"\nModel type: {'Skip-gram' if config['sg'] == 1 else 'CBOW'}")

    start_time = time.time()

    model = Word2Vec(
        sentences=sentences,
        vector_size=config['vector_size'],
        window=config['window'],
        min_count=config['min_count'],
        workers=config['workers'],
        sg=config['sg'],
        negative=config['negative'],
        sample=config['sample'],
        alpha=config['alpha'],
        min_alpha=config['min_alpha'],
        epochs=config['epochs']
    )

    training_time = time.time() - start_time

    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Vocabulary size: {len(model.wv)}")
    print(f"Vector size: {model.wv.vector_size}")

    return model

def load_glove_embeddings():
    """Load pre-trained GloVe embeddings"""
    print("\n" + "="*80)
    print("LOADING PRE-TRAINED GLOVE EMBEDDINGS")
    print("="*80)

    # Note: This is a placeholder. In practice, download GloVe from:
    # https://nlp.stanford.edu/projects/glove/

    print("\nTo use GloVe embeddings:")
    print("1. Download from: https://nlp.stanford.edu/projects/glove/")
    print("2. Extract glove.6B.100d.txt (or other dimension)")
    print("3. Load with gensim:")
    print("   from gensim.scripts.glove2word2vec import glove2word2vec")
    print("   glove2word2vec('glove.6B.100d.txt', 'glove.word2vec.txt')")
    print("   model = KeyedVectors.load_word2vec_format('glove.word2vec.txt')")

    # Try to load if available
    glove_file = Path('data/glove.6B.100d.txt')
    w2v_file = Path('data/glove.word2vec.txt')

    if glove_file.exists():
        print(f"\nFound GloVe file: {glove_file}")

        if not w2v_file.exists():
            print("Converting GloVe to Word2Vec format...")
            from gensim.scripts.glove2word2vec import glove2word2vec
            glove2word2vec(str(glove_file), str(w2v_file))

        print("Loading GloVe embeddings...")
        glove_model = KeyedVectors.load_word2vec_format(str(w2v_file))
        print(f"Loaded GloVe with vocabulary size: {len(glove_model)}")

        return glove_model
    else:
        print(f"\nGloVe file not found at {glove_file}")
        print("Continuing with Word2Vec model only...")
        return None

def explore_embeddings(model, model_name="Word2Vec"):
    """Explore word embeddings"""
    print("\n" + "="*80)
    print(f"EXPLORING {model_name} EMBEDDINGS")
    print("="*80)

    # Get word vectors
    wv = model.wv if hasattr(model, 'wv') else model

    # Test words
    test_words = ['king', 'queen', 'man', 'woman', 'computer', 'science']

    print("\n1. WORD SIMILARITIES")
    print("-" * 80)

    for word in test_words:
        if word in wv:
            similar_words = wv.most_similar(word, topn=5)
            print(f"\nMost similar to '{word}':")
            for similar_word, score in similar_words:
                print(f"  {similar_word:15} {score:.4f}")
        else:
            print(f"\nWord '{word}' not in vocabulary")

    # Word analogies
    print("\n2. WORD ANALOGIES (A is to B as C is to ?)")
    print("-" * 80)

    analogies = [
        ('king', 'man', 'queen'),  # king - man + woman ≈ queen
        ('paris', 'france', 'london'),  # paris - france + england ≈ london
        ('good', 'better', 'bad'),  # good - better + bad ≈ worse
    ]

    for word_a, word_b, word_c in analogies:
        try:
            if all(w in wv for w in [word_a, word_b, word_c]):
                result = wv.most_similar(
                    positive=[word_b, word_c],
                    negative=[word_a],
                    topn=3
                )
                print(f"\n'{word_a}' is to '{word_b}' as '{word_c}' is to:")
                for word, score in result:
                    print(f"  {word:15} {score:.4f}")
            else:
                print(f"\nSkipping analogy (words not in vocabulary): {word_a}, {word_b}, {word_c}")
        except Exception as e:
            print(f"\nError computing analogy: {e}")

    # Odd one out
    print("\n3. ODD ONE OUT")
    print("-" * 80)

    word_lists = [
        ['breakfast', 'cereal', 'dinner', 'lunch'],
        ['dog', 'cat', 'mouse', 'furniture'],
    ]

    for words in word_lists:
        try:
            if all(w in wv for w in words):
                odd_one = wv.doesnt_match(words)
                print(f"\nWords: {words}")
                print(f"Odd one out: {odd_one}")
            else:
                missing = [w for w in words if w not in wv]
                print(f"\nSkipping (words not in vocabulary): {missing}")
        except Exception as e:
            print(f"\nError: {e}")

    # Similarity scores
    print("\n4. SIMILARITY SCORES")
    print("-" * 80)

    word_pairs = [
        ('king', 'queen'),
        ('man', 'woman'),
        ('computer', 'keyboard'),
        ('car', 'automobile'),
        ('king', 'car'),
    ]

    for word1, word2 in word_pairs:
        if word1 in wv and word2 in wv:
            similarity = wv.similarity(word1, word2)
            print(f"Similarity('{word1}', '{word2}'): {similarity:.4f}")
        else:
            print(f"Skipping pair (not in vocabulary): {word1}, {word2}")

def visualize_embeddings(model, words=None, method='tsne'):
    """Visualize word embeddings in 2D"""
    print("\n" + "="*80)
    print("VISUALIZING EMBEDDINGS")
    print("="*80)

    wv = model.wv if hasattr(model, 'wv') else model

    # Select words to visualize
    if words is None:
        # Get most frequent words
        words = [word for word in list(wv.index_to_key)[:100]]

    # Filter words that exist in vocabulary
    words = [w for w in words if w in wv]

    if len(words) == 0:
        print("No valid words to visualize")
        return

    print(f"\nVisualizing {len(words)} words using {method.upper()}")

    # Get word vectors
    word_vectors = np.array([wv[word] for word in words])

    # Dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=2)
    else:  # tsne
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(words)-1))

    reduced_vectors = reducer.fit_transform(word_vectors)

    # Plot
    plt.figure(figsize=(14, 10))
    plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], alpha=0.5)

    # Annotate words
    for i, word in enumerate(words):
        plt.annotate(
            word,
            xy=(reduced_vectors[i, 0], reduced_vectors[i, 1]),
            xytext=(5, 2),
            textcoords='offset points',
            ha='right',
            fontsize=8,
            alpha=0.7
        )

    plt.title(f'Word Embeddings Visualization ({method.upper()})')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True, alpha=0.3)

    # Save plot
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f'word_embeddings_{method}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")

    plt.close()

def save_model(model, model_name='word2vec'):
    """Save trained model"""
    output_dir = Path('models')
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f'{model_name}.model'
    model.save(str(output_file))
    print(f"\nModel saved to: {output_file}")

    # Save word vectors only (smaller file)
    wv_file = output_dir / f'{model_name}.wordvectors'
    model.wv.save(str(wv_file))
    print(f"Word vectors saved to: {wv_file}")

def main():
    print("="*80)
    print("WORD EMBEDDINGS WITH GENSIM")
    print("="*80)

    # Download and load corpus
    corpus_file = download_text8()

    if corpus_file is None:
        print("\nCould not load corpus. Please download text8 manually.")
        print("URL: http://mattmahoney.net/dc/text8.zip")
        return

    sentences = load_corpus(corpus_file)

    # Limit sentences for faster training (demo purpose)
    max_sentences = 10000
    if len(sentences) > max_sentences:
        print(f"\nLimiting to {max_sentences} sentences for demo...")
        sentences = sentences[:max_sentences]

    # Train Word2Vec
    model = train_word2vec(sentences, CONFIG)

    # Explore embeddings
    explore_embeddings(model, "Word2Vec")

    # Try to load GloVe
    glove_model = load_glove_embeddings()
    if glove_model is not None:
        explore_embeddings(glove_model, "GloVe")

    # Visualize embeddings
    visualization_words = [
        'man', 'woman', 'king', 'queen',
        'computer', 'science', 'technology',
        'cat', 'dog', 'animal',
        'good', 'bad', 'better', 'worse',
        'run', 'walk', 'jump'
    ]

    visualize_embeddings(model, words=visualization_words, method='tsne')
    visualize_embeddings(model, words=visualization_words, method='pca')

    # Save model
    save_model(model, 'word2vec_text8')

    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)

    print("\nModel Summary:")
    print(f"- Algorithm: {'Skip-gram' if CONFIG['sg'] == 1 else 'CBOW'}")
    print(f"- Vocabulary size: {len(model.wv)}")
    print(f"- Vector dimension: {CONFIG['vector_size']}")
    print(f"- Context window: {CONFIG['window']}")
    print(f"- Training epochs: {CONFIG['epochs']}")

    print("\nKey Concepts:")
    print("✓ Word2Vec learns distributed representations of words")
    print("✓ Similar words have similar vector representations")
    print("✓ Can capture semantic relationships (king - man + woman ≈ queen)")
    print("✓ Two training algorithms: CBOW (fast) and Skip-gram (better for rare words)")
    print("✓ Negative sampling improves training efficiency")

    print("\nUse Cases:")
    print("- Text classification feature extraction")
    print("- Semantic similarity and clustering")
    print("- Information retrieval")
    print("- Machine translation")
    print("- Named entity recognition")
    print("- Sentiment analysis")

if __name__ == '__main__':
    main()
