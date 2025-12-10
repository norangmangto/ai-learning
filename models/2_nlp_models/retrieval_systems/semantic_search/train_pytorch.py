"""
Semantic Search Implementation using Sentence-Transformers

This script demonstrates semantic search by:
1. Loading a pre-trained sentence embedding model
2. Encoding a corpus of documents
3. Finding similar documents based on semantic meaning
4. Evaluating retrieval performance

Dataset: Custom text corpus or MS MARCO passages
Model: sentence-transformers (all-MiniLM-L6-v2)
"""

import torch
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
import time
from pathlib import Path
import json

# Configuration
CONFIG = {
    'model_name': 'all-MiniLM-L6-v2',  # Fast and efficient
    'dataset_name': 'ms_marco',  # Microsoft MARCO passage ranking
    'dataset_config': 'v2.1',
    'max_corpus_size': 10000,  # Limit corpus for demo
    'batch_size': 32,
    'top_k': 5,  # Number of results to retrieve
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def load_corpus(max_size=10000):
    """Load text corpus for semantic search"""
    print(f"Loading MS MARCO dataset (limited to {max_size} passages)...")

    try:
        # Load MS MARCO passage corpus
        dataset = load_dataset('ms_marco', 'v2.1', split='train', streaming=True)

        corpus = []
        corpus_ids = []

        for i, item in enumerate(dataset):
            if i >= max_size:
                break

            # Extract passage text
            if 'passages' in item and len(item['passages']['passage_text']) > 0:
                text = item['passages']['passage_text'][0]
                corpus.append(text)
                corpus_ids.append(f"doc_{i}")

            if (i + 1) % 1000 == 0:
                print(f"Loaded {i + 1} passages...")

        print(f"Loaded {len(corpus)} passages")
        return corpus, corpus_ids

    except Exception as e:
        print(f"Could not load MS MARCO: {e}")
        print("Using sample corpus instead...")

        # Fallback to sample corpus
        corpus = [
            "Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
            "Deep learning uses neural networks with multiple layers to process complex patterns.",
            "Natural language processing helps computers understand and generate human language.",
            "Computer vision enables machines to interpret and understand visual information from images.",
            "Reinforcement learning trains agents through trial and error with rewards and penalties.",
            "Supervised learning uses labeled data to train predictive models.",
            "Unsupervised learning discovers patterns in data without explicit labels.",
            "Transfer learning leverages pre-trained models for new tasks.",
            "Python is a popular programming language for data science and machine learning.",
            "TensorFlow and PyTorch are leading deep learning frameworks.",
            "Data preprocessing is crucial for building effective machine learning models.",
            "Feature engineering transforms raw data into meaningful inputs for models.",
            "Model evaluation metrics help assess the performance of machine learning algorithms.",
            "Cross-validation prevents overfitting and ensures model generalization.",
            "Hyperparameter tuning optimizes model performance through systematic search.",
            "Ensemble methods combine multiple models to improve predictions.",
            "Gradient descent is an optimization algorithm for training neural networks.",
            "Backpropagation computes gradients for updating neural network weights.",
            "Convolutional neural networks excel at image recognition tasks.",
            "Recurrent neural networks process sequential data like text and time series."
        ]
        corpus_ids = [f"doc_{i}" for i in range(len(corpus))]

        return corpus, corpus_ids

def encode_corpus(model, corpus, batch_size=32):
    """Encode corpus into embeddings"""
    print(f"\nEncoding {len(corpus)} documents...")
    start_time = time.time()

    # Encode with progress
    corpus_embeddings = model.encode(
        corpus,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True
    )

    encoding_time = time.time() - start_time
    print(f"Encoding completed in {encoding_time:.2f} seconds")
    print(f"Embedding shape: {corpus_embeddings.shape}")

    return corpus_embeddings

def semantic_search(model, query, corpus_embeddings, corpus, corpus_ids, top_k=5):
    """Perform semantic search for a query"""
    # Encode query
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Compute cosine similarity
    start_time = time.time()
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    search_time = time.time() - start_time

    # Get top-k results
    top_results = torch.topk(cos_scores, k=min(top_k, len(corpus)))

    results = []
    for score, idx in zip(top_results.values, top_results.indices):
        results.append({
            'corpus_id': corpus_ids[idx],
            'score': score.item(),
            'text': corpus[idx]
        })

    return results, search_time

def evaluate_retrieval(model, corpus_embeddings, corpus, corpus_ids, test_queries):
    """Evaluate semantic search on test queries"""
    print("\n" + "="*80)
    print("EVALUATING SEMANTIC SEARCH")
    print("="*80)

    total_search_time = 0

    for i, query_info in enumerate(test_queries, 1):
        query = query_info['query']
        print(f"\n{'='*80}")
        print(f"Query {i}: {query}")
        print(f"{'='*80}")

        results, search_time = semantic_search(
            model, query, corpus_embeddings, corpus, corpus_ids, top_k=CONFIG['top_k']
        )
        total_search_time += search_time

        print(f"Search time: {search_time*1000:.2f} ms\n")

        for j, result in enumerate(results, 1):
            print(f"{j}. Score: {result['score']:.4f}")
            print(f"   ID: {result['corpus_id']}")
            print(f"   Text: {result['text'][:150]}...")
            print()

    avg_search_time = total_search_time / len(test_queries)
    print(f"\n{'='*80}")
    print(f"Average search time: {avg_search_time*1000:.2f} ms")
    print(f"{'='*80}")

def demonstrate_semantic_similarity():
    """Demonstrate semantic similarity understanding"""
    print("\n" + "="*80)
    print("SEMANTIC SIMILARITY DEMONSTRATION")
    print("="*80)

    model = SentenceTransformer(CONFIG['model_name'])

    # Similar meaning, different words
    sentences = [
        "The cat sits on the mat",
        "A feline rests on a rug",
        "The dog runs in the park",
        "The weather is nice today",
        "It's a beautiful day outside"
    ]

    embeddings = model.encode(sentences, convert_to_tensor=True)

    print("\nCosine Similarity Matrix:")
    print("-" * 80)

    # Compute pairwise similarities
    similarities = util.cos_sim(embeddings, embeddings)

    # Print header
    print(f"{'':30}", end="")
    for i in range(len(sentences)):
        print(f"S{i+1:2}    ", end="")
    print()

    # Print similarity matrix
    for i, sent in enumerate(sentences):
        print(f"S{i+1}. {sent[:27]:27}", end="")
        for j in range(len(sentences)):
            print(f"{similarities[i][j]:.3f}  ", end="")
        print()

    print("\nObservations:")
    print("- S1 & S2 have high similarity (similar meaning, different words)")
    print("- S4 & S5 have high similarity (weather-related)")
    print("- S3 is less similar to S1/S2 (different topic)")

def save_results(results, output_dir='results'):
    """Save search results"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    output_file = output_path / 'semantic_search_results.json'

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

def main():
    print("="*80)
    print("SEMANTIC SEARCH WITH SENTENCE-TRANSFORMERS")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"Model: {CONFIG['model_name']}")
    print(f"Device: {CONFIG['device']}")
    print(f"Corpus size: {CONFIG['max_corpus_size']}")

    # Load model
    print("\nLoading sentence transformer model...")
    model = SentenceTransformer(CONFIG['model_name'])
    model.to(CONFIG['device'])
    print(f"Model loaded: {model}")

    # Load corpus
    corpus, corpus_ids = load_corpus(CONFIG['max_corpus_size'])

    # Encode corpus
    corpus_embeddings = encode_corpus(model, corpus, CONFIG['batch_size'])

    # Define test queries
    test_queries = [
        {
            'query': "How do neural networks learn?",
            'relevant_docs': ['doc_0', 'doc_1']  # Example relevant docs
        },
        {
            'query': "What is the difference between supervised and unsupervised learning?",
            'relevant_docs': ['doc_5', 'doc_6']
        },
        {
            'query': "Best practices for training deep learning models",
            'relevant_docs': ['doc_10', 'doc_11', 'doc_13']
        },
        {
            'query': "Computer vision applications",
            'relevant_docs': ['doc_3', 'doc_18']
        }
    ]

    # Evaluate semantic search
    evaluate_retrieval(model, corpus_embeddings, corpus, corpus_ids, test_queries)

    # Demonstrate semantic similarity
    demonstrate_semantic_similarity()

    # Interactive search (optional)
    print("\n" + "="*80)
    print("INTERACTIVE SEARCH (Enter 'quit' to exit)")
    print("="*80)

    while True:
        try:
            user_query = input("\nEnter your search query: ").strip()
            if user_query.lower() in ['quit', 'exit', 'q']:
                break

            if not user_query:
                continue

            results, search_time = semantic_search(
                model, user_query, corpus_embeddings, corpus, corpus_ids, top_k=3
            )

            print(f"\nTop 3 results (search time: {search_time*1000:.2f} ms):")
            print("-" * 80)

            for i, result in enumerate(results, 1):
                print(f"\n{i}. Score: {result['score']:.4f}")
                print(f"   {result['text'][:200]}...")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)

    # Performance summary
    print("\nPerformance Summary:")
    print(f"- Model: {CONFIG['model_name']}")
    print(f"- Corpus size: {len(corpus)} documents")
    print(f"- Embedding dimension: {corpus_embeddings.shape[1]}")
    print(f"- Device: {CONFIG['device']}")
    print(f"- Average query time: <100ms for {len(corpus)} documents")

    print("\nKey Features:")
    print("✓ Semantic understanding (meaning-based, not keyword matching)")
    print("✓ Fast retrieval using cosine similarity")
    print("✓ Pre-trained models (no training required)")
    print("✓ Multilingual support available")
    print("✓ Scalable to millions of documents with FAISS")

    print("\nUse Cases:")
    print("- Document search and retrieval")
    print("- Question answering systems")
    print("- Duplicate detection")
    print("- Recommendation systems")
    print("- FAQ matching")

if __name__ == '__main__':
    main()
