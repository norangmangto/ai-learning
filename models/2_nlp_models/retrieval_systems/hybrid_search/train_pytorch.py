"""
Hybrid Search System (BM25 + Semantic Search)

This script demonstrates:
1. Lexical search with BM25 (keyword matching)
2. Semantic search with embeddings (meaning-based)
3. Hybrid fusion of both approaches
4. Reciprocal Rank Fusion (RRF)
5. Comparing different retrieval strategies

Combines: BM25 + Sentence-BERT
Best of both worlds: keyword precision + semantic understanding
"""

import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from datasets import load_dataset
import time
from collections import defaultdict

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Configuration
CONFIG = {
    'semantic_model': 'all-MiniLM-L6-v2',
    'corpus_size': 1000,
    'top_k': 10,
    'alpha': 0.5,  # Hybrid weight (0=BM25, 1=semantic)
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'output_dir': 'results/hybrid_search'
}

class HybridSearchSystem:
    """Hybrid search combining BM25 and semantic search"""

    def __init__(self, corpus, semantic_model_name='all-MiniLM-L6-v2', device='cpu'):
        self.corpus = corpus
        self.device = device

        print("Initializing Hybrid Search System...")
        print("=" * 80)

        # BM25 setup
        print("\n1. Setting up BM25 (lexical search)...")
        self.tokenized_corpus = [self._tokenize(doc) for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"   BM25 index built for {len(corpus)} documents")

        # Semantic search setup
        print("\n2. Setting up semantic search...")
        self.semantic_model = SentenceTransformer(semantic_model_name)
        self.semantic_model.to(device)
        print(f"   Loading model: {semantic_model_name}")

        print("\n3. Encoding corpus...")
        self.corpus_embeddings = self.semantic_model.encode(
            corpus,
            convert_to_tensor=True,
            show_progress_bar=True
        )
        print(f"   Corpus embeddings shape: {self.corpus_embeddings.shape}")

        print("\nHybrid Search System ready!")
        print("=" * 80)

    def _tokenize(self, text):
        """Tokenize text for BM25"""
        try:
            tokens = word_tokenize(text.lower())
            # Remove stopwords and short tokens
            stop_words = set(stopwords.words('english'))
            tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
            return tokens
        except:
            return text.lower().split()

    def search_bm25(self, query, top_k=10):
        """BM25 lexical search"""
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                'idx': int(idx),
                'score': float(scores[idx]),
                'text': self.corpus[idx]
            })

        return results

    def search_semantic(self, query, top_k=10):
        """Semantic search with embeddings"""
        query_embedding = self.semantic_model.encode(
            query,
            convert_to_tensor=True
        )

        # Cosine similarity
        cos_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]

        # Get top-k indices
        top_results = torch.topk(cos_scores, k=min(top_k, len(self.corpus)))

        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            results.append({
                'idx': int(idx),
                'score': float(score),
                'text': self.corpus[idx]
            })

        return results

    def search_hybrid(self, query, top_k=10, alpha=0.5):
        """
        Hybrid search combining BM25 and semantic search

        Args:
            query: Search query
            top_k: Number of results
            alpha: Weight (0=only BM25, 1=only semantic, 0.5=balanced)
        """
        # Get results from both methods
        bm25_results = self.search_bm25(query, top_k=top_k*2)
        semantic_results = self.search_semantic(query, top_k=top_k*2)

        # Normalize scores
        if len(bm25_results) > 0:
            max_bm25 = max(r['score'] for r in bm25_results)
            min_bm25 = min(r['score'] for r in bm25_results)
            if max_bm25 > min_bm25:
                for r in bm25_results:
                    r['score'] = (r['score'] - min_bm25) / (max_bm25 - min_bm25)

        # Combine scores
        combined_scores = defaultdict(float)

        for r in bm25_results:
            combined_scores[r['idx']] += (1 - alpha) * r['score']

        for r in semantic_results:
            combined_scores[r['idx']] += alpha * r['score']

        # Sort by combined score
        sorted_indices = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        results = []
        for idx, score in sorted_indices:
            results.append({
                'idx': idx,
                'score': score,
                'text': self.corpus[idx]
            })

        return results

    def search_rrf(self, query, top_k=10, k=60):
        """
        Reciprocal Rank Fusion (RRF)

        Combines rankings from BM25 and semantic search
        RRF score = sum(1 / (k + rank))
        """
        # Get results from both methods
        bm25_results = self.search_bm25(query, top_k=top_k*2)
        semantic_results = self.search_semantic(query, top_k=top_k*2)

        # Build rank dictionaries
        bm25_ranks = {r['idx']: rank+1 for rank, r in enumerate(bm25_results)}
        semantic_ranks = {r['idx']: rank+1 for rank, r in enumerate(semantic_results)}

        # Compute RRF scores
        all_indices = set(bm25_ranks.keys()) | set(semantic_ranks.keys())
        rrf_scores = {}

        for idx in all_indices:
            score = 0
            if idx in bm25_ranks:
                score += 1 / (k + bm25_ranks[idx])
            if idx in semantic_ranks:
                score += 1 / (k + semantic_ranks[idx])
            rrf_scores[idx] = score

        # Sort by RRF score
        sorted_indices = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        results = []
        for idx, score in sorted_indices:
            results.append({
                'idx': idx,
                'score': score,
                'text': self.corpus[idx]
            })

        return results

def load_corpus(max_size=1000):
    """Load document corpus"""
    print("="*80)
    print("LOADING CORPUS")
    print("="*80)

    try:
        # Load MS MARCO passages
        dataset = load_dataset('ms_marco', 'v2.1', split='train', streaming=True)

        corpus = []
        for i, item in enumerate(dataset):
            if i >= max_size:
                break

            if 'passages' in item and len(item['passages']['passage_text']) > 0:
                text = item['passages']['passage_text'][0]
                corpus.append(text)

        print(f"Loaded {len(corpus)} documents from MS MARCO")
        return corpus

    except Exception as e:
        print(f"Could not load MS MARCO: {e}")
        print("Using sample corpus...")

        # Sample corpus
        corpus = [
            "Machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming.",
            "Deep learning uses neural networks with multiple layers to automatically learn hierarchical representations of data.",
            "Natural language processing helps computers understand, interpret, and generate human language.",
            "Computer vision enables machines to extract meaningful information from digital images and videos.",
            "Reinforcement learning trains agents to make sequences of decisions by maximizing cumulative rewards.",
            "Supervised learning uses labeled training data to learn a mapping from inputs to outputs.",
            "Unsupervised learning discovers hidden patterns and structures in unlabeled data.",
            "Transfer learning leverages knowledge from pre-trained models for new related tasks.",
            "Python is the most popular programming language for machine learning and data science applications.",
            "TensorFlow and PyTorch are the leading frameworks for building and training deep learning models.",
            "Neural networks are inspired by the structure and function of biological neurons in the brain.",
            "Gradient descent is an optimization algorithm used to minimize loss functions during training.",
            "Convolutional neural networks are specifically designed for processing grid-like data such as images.",
            "Recurrent neural networks process sequential data by maintaining hidden states across time steps.",
            "Transformers use self-attention mechanisms to process sequences in parallel for better efficiency.",
            "Autoencoders learn compressed representations of data through unsupervised learning.",
            "Generative adversarial networks consist of two neural networks competing against each other.",
            "Feature engineering transforms raw data into meaningful inputs for machine learning models.",
            "Cross-validation is a technique for assessing model generalization on unseen data.",
            "Hyperparameter tuning optimizes model configuration for better performance on validation data."
        ]

        return corpus

def compare_search_methods(hybrid_system, queries):
    """Compare different search methods"""
    print("\n" + "="*80)
    print("COMPARING SEARCH METHODS")
    print("="*80)

    for i, query in enumerate(queries, 1):
        print(f"\n{'='*80}")
        print(f"Query {i}: {query}")
        print(f"{'='*80}")

        # BM25
        print("\n1. BM25 (Lexical Search)")
        print("-" * 80)
        start = time.time()
        bm25_results = hybrid_system.search_bm25(query, top_k=3)
        bm25_time = time.time() - start

        for j, result in enumerate(bm25_results[:3], 1):
            print(f"{j}. Score: {result['score']:.4f}")
            print(f"   {result['text'][:100]}...")
        print(f"Time: {bm25_time*1000:.2f}ms")

        # Semantic
        print("\n2. Semantic Search")
        print("-" * 80)
        start = time.time()
        semantic_results = hybrid_system.search_semantic(query, top_k=3)
        semantic_time = time.time() - start

        for j, result in enumerate(semantic_results[:3], 1):
            print(f"{j}. Score: {result['score']:.4f}")
            print(f"   {result['text'][:100]}...")
        print(f"Time: {semantic_time*1000:.2f}ms")

        # Hybrid
        print("\n3. Hybrid Search (α=0.5)")
        print("-" * 80)
        start = time.time()
        hybrid_results = hybrid_system.search_hybrid(query, top_k=3, alpha=0.5)
        hybrid_time = time.time() - start

        for j, result in enumerate(hybrid_results[:3], 1):
            print(f"{j}. Score: {result['score']:.4f}")
            print(f"   {result['text'][:100]}...")
        print(f"Time: {hybrid_time*1000:.2f}ms")

        # RRF
        print("\n4. Reciprocal Rank Fusion (RRF)")
        print("-" * 80)
        start = time.time()
        rrf_results = hybrid_system.search_rrf(query, top_k=3)
        rrf_time = time.time() - start

        for j, result in enumerate(rrf_results[:3], 1):
            print(f"{j}. Score: {result['score']:.4f}")
            print(f"   {result['text'][:100]}...")
        print(f"Time: {rrf_time*1000:.2f}ms")

def main():
    print("="*80)
    print("HYBRID SEARCH SYSTEM")
    print("BM25 + Semantic Search")
    print("="*80)

    print(f"\nDevice: {CONFIG['device']}")

    # Load corpus
    corpus = load_corpus(CONFIG['corpus_size'])

    # Build hybrid search system
    hybrid_system = HybridSearchSystem(
        corpus,
        CONFIG['semantic_model'],
        CONFIG['device']
    )

    # Test queries
    test_queries = [
        "How do neural networks learn?",
        "What is the difference between supervised and unsupervised learning?",
        "Explain transformers in deep learning",
        "Python frameworks for machine learning"
    ]

    # Compare methods
    compare_search_methods(hybrid_system, test_queries)

    print("\n" + "="*80)
    print("HYBRID SEARCH COMPLETED")
    print("="*80)

    print("\nSearch Method Comparison:")
    print("\n1. BM25 (Lexical)")
    print("   + Fast and efficient")
    print("   + Good for keyword/exact matches")
    print("   + No ML model needed")
    print("   - Misses semantic meaning")
    print("   - Vocabulary mismatch problem")

    print("\n2. Semantic Search")
    print("   + Understands meaning")
    print("   + Handles paraphrases")
    print("   + Cross-lingual capability")
    print("   - May miss exact keywords")
    print("   - Requires embeddings")

    print("\n3. Hybrid (Best of Both)")
    print("   + Combines precision of BM25")
    print("   + With semantic understanding")
    print("   + Tunable with alpha parameter")
    print("   + Better overall performance")

    print("\nWhen to Use Each:")
    print("- BM25: Keyword-heavy queries, exact matches")
    print("- Semantic: Conceptual queries, paraphrases")
    print("- Hybrid: General purpose, production systems")
    print("- RRF: When you want rank-based fusion")

    print("\nApplications:")
    print("- Document search and retrieval")
    print("- Question answering systems")
    print("- E-commerce product search")
    print("- Enterprise knowledge bases")
    print("- Legal document discovery")

if __name__ == '__main__':
    main()
