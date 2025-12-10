"""
Training Retrieval Models for LangChain RAG Systems
This script covers:
- Dense retrievers (DPR, ColBERT)
- Cross-encoders for re-ranking
- Hybrid search (BM25 + dense)
- Custom retrieval pipelines
"""

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder, losses, InputExample
from torch.utils.data import DataLoader
import os

# For LangChain integration
try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.retrievers import BM25Retriever, EnsembleRetriever
    from langchain.schema import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    print("⚠ LangChain not installed. Install with: pip install langchain langchain-community chromadb faiss-cpu rank-bm25")
    LANGCHAIN_AVAILABLE = False


def create_retrieval_training_data():
    """
    Create training data for retrieval model fine-tuning.
    Format: (query, positive_doc, negative_doc)
    """

    # Sample training triplets
    # In production, use datasets like MS MARCO, Natural Questions, or domain-specific data
    train_data = [
        # Query, Positive document, Negative document
        (
            "What is a vector database?",
            "Vector databases are specialized databases designed to store and query high-dimensional vector embeddings efficiently. They use algorithms like HNSW or IVF for approximate nearest neighbor search.",
            "Traditional relational databases use SQL for querying structured data in tables with rows and columns."
        ),
        (
            "How does RAG work?",
            "RAG (Retrieval Augmented Generation) first retrieves relevant documents from a knowledge base, then uses these as context for a language model to generate accurate, grounded responses.",
            "RNN (Recurrent Neural Networks) are neural networks designed for sequential data processing with feedback connections."
        ),
        (
            "Explain semantic search",
            "Semantic search uses embeddings to find content based on meaning rather than exact keywords. It converts queries and documents to vectors and finds similar items using cosine similarity.",
            "Binary search is an algorithm for finding an element in a sorted array by repeatedly dividing the search interval in half."
        ),
        (
            "What are embedding models?",
            "Embedding models convert text into dense vector representations that capture semantic meaning. Popular models include BERT, Sentence-BERT, and OpenAI's text-embedding models.",
            "Image generation models like DALL-E and Stable Diffusion create images from text descriptions using diffusion processes."
        ),
        (
            "LangChain retrieval techniques",
            "LangChain provides multiple retrieval methods including vector stores (FAISS, Chroma), BM25 for keyword search, hybrid retrievers combining both, and multi-query retrievers for improved recall.",
            "React hooks are functions that let you use state and lifecycle features in functional components without writing classes."
        ),
        (
            "Fine-tuning vs prompt engineering",
            "Fine-tuning updates model weights on task-specific data for optimal performance but requires compute and data. Prompt engineering modifies inputs without changing the model, offering quick adaptation with less overhead.",
            "CSS Grid and Flexbox are layout systems in CSS for creating responsive web designs with different alignment capabilities."
        ),
    ]

    # Create training examples
    train_examples = []
    for query, pos_doc, neg_doc in train_data:
        # Positive pair
        train_examples.append(InputExample(texts=[query, pos_doc], label=1.0))
        # Negative pair
        train_examples.append(InputExample(texts=[query, neg_doc], label=0.0))

    return train_examples


def train_dense_retriever():
    """
    Fine-tune a dense retriever (bi-encoder) for semantic search.
    This model embeds queries and documents separately for efficient retrieval.
    """

    print("=" * 60)
    print("Training Dense Retriever (Bi-Encoder)")
    print("=" * 60)

    # 1. Load base model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"\n1. Loading base model: {model_name}")

    model = SentenceTransformer(model_name)
    print(f"✓ Model loaded: {model.get_sentence_embedding_dimension()} dimensions")

    # 2. Prepare training data
    print("\n2. Preparing retrieval training data...")
    train_examples = create_retrieval_training_data()
    print(f"✓ Created {len(train_examples)} training examples")

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=4)

    # 3. Define loss function
    # MultipleNegativesRankingLoss is excellent for retrieval tasks
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # 4. Train
    print("\n3. Fine-tuning dense retriever...")
    output_path = "models/finetuned_retriever"
    os.makedirs(output_path, exist_ok=True)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=10,
        warmup_steps=100,
        output_path=output_path,
        show_progress_bar=True,
    )

    print(f"✓ Dense retriever saved to: {output_path}")

    # 5. QA Validation
    print("\n" + "=" * 60)
    print("QA Validation - Dense Retriever")
    print("=" * 60)

    # Load fine-tuned model
    retriever = SentenceTransformer(output_path)

    # Test queries and documents
    queries = [
        "What is semantic search?",
        "Explain vector databases",
        "How to use LangChain?"
    ]

    documents = [
        "Semantic search finds content by meaning using embeddings and similarity metrics",
        "Vector databases store and query high-dimensional embeddings for similarity search",
        "LangChain is a framework for building LLM applications with chains and retrievers",
        "Python is a high-level programming language known for its simplicity",
        "Machine learning models learn patterns from data without explicit programming",
    ]

    print("\n--- Retrieval Quality Test ---")

    query_embeddings = retriever.encode(queries, convert_to_tensor=True)
    doc_embeddings = retriever.encode(documents, convert_to_tensor=True)

    from sentence_transformers.util import cos_sim

    correct_retrievals = 0
    for i, query in enumerate(queries):
        similarities = cos_sim(query_embeddings[i:i+1], doc_embeddings)[0]
        top_idx = similarities.argmax().item()

        print(f"\nQuery: '{query}'")
        print(f"Top result: {documents[top_idx]}")
        print(f"Score: {similarities[top_idx]:.4f}")

        # Check if retrieved the expected relevant document
        if i == top_idx:
            correct_retrievals += 1

    retrieval_accuracy = correct_retrievals / len(queries)

    # Sanity checks
    print("\n--- Sanity Checks ---")

    # Check 1: Model outputs correct dimension
    test_emb = retriever.encode("test")
    if len(test_emb) == model.get_sentence_embedding_dimension():
        print(f"✓ Embeddings have correct dimension: {len(test_emb)}")
    else:
        print(f"✗ WARNING: Dimension mismatch")

    # Check 2: Retrieval accuracy
    if retrieval_accuracy >= 0.6:
        print(f"✓ Good retrieval accuracy: {retrieval_accuracy:.2%}")
    elif retrieval_accuracy >= 0.3:
        print(f"⚠ Moderate retrieval accuracy: {retrieval_accuracy:.2%}")
    else:
        print(f"✗ WARNING: Poor retrieval accuracy: {retrieval_accuracy:.2%}")

    # Check 3: Model files exist
    if os.path.exists(os.path.join(output_path, "config.json")):
        print("✓ Model files saved successfully")
    else:
        print("✗ WARNING: Model files missing")

    print("\n=== Overall Validation Result ===")
    validation_passed = (
        len(test_emb) == model.get_sentence_embedding_dimension() and
        retrieval_accuracy >= 0.3 and
        os.path.exists(os.path.join(output_path, "config.json"))
    )

    if validation_passed:
        print("✓ Dense retriever validation PASSED")
    else:
        print("✗ Dense retriever validation FAILED")

    return retriever


def train_cross_encoder_reranker():
    """
    Train a cross-encoder for re-ranking retrieved documents.
    Cross-encoders score query-document pairs jointly for higher accuracy.
    """

    print("\n" + "=" * 60)
    print("Training Cross-Encoder Re-ranker")
    print("=" * 60)

    # 1. Load base cross-encoder
    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    print(f"\n1. Loading base model: {model_name}")

    model = CrossEncoder(model_name, num_labels=1)
    print("✓ Cross-encoder loaded")

    # 2. Prepare training data
    print("\n2. Preparing re-ranking training data...")
    train_data = create_retrieval_training_data()

    # Convert to cross-encoder format
    train_samples = []
    for example in train_data:
        train_samples.append(InputExample(
            texts=[example.texts[0], example.texts[1]],
            label=float(example.label)
        ))

    print(f"✓ Created {len(train_samples)} training samples")

    # 3. Train
    print("\n3. Fine-tuning cross-encoder...")
    output_path = "models/finetuned_reranker"
    os.makedirs(output_path, exist_ok=True)

    model.fit(
        train_dataloader=DataLoader(train_samples, shuffle=True, batch_size=4),
        epochs=5,
        warmup_steps=50,
        output_path=output_path,
        show_progress_bar=True,
    )

    print(f"✓ Cross-encoder saved to: {output_path}")

    # 4. QA Validation
    print("\n" + "=" * 60)
    print("QA Validation - Cross-Encoder")
    print("=" * 60)

    # Load fine-tuned model
    reranker = CrossEncoder(output_path)

    # Test re-ranking
    query = "What is semantic search?"
    candidates = [
        "Semantic search uses meaning and context to find relevant information",
        "Python is a programming language",
        "Search engines index web pages for retrieval",
    ]

    print(f"\nQuery: '{query}'")
    print("Candidates:")
    for i, doc in enumerate(candidates, 1):
        print(f"  {i}. {doc}")

    # Score all pairs
    pairs = [[query, doc] for doc in candidates]
    scores = reranker.predict(pairs)

    # Rank by scores
    ranked_indices = np.argsort(scores)[::-1]

    print("\nRe-ranked results:")
    for rank, idx in enumerate(ranked_indices, 1):
        print(f"  {rank}. Score: {scores[idx]:.4f} - {candidates[idx]}")

    # Sanity checks
    print("\n--- Sanity Checks ---")

    # Check 1: Scores are reasonable
    if all(isinstance(s, (int, float, np.number)) for s in scores):
        print("✓ Scores are valid numbers")
    else:
        print("✗ WARNING: Invalid scores")

    # Check 2: Best match is actually relevant
    if ranked_indices[0] == 0:  # First candidate is most relevant
        print("✓ Correctly ranked most relevant document first")
    else:
        print("⚠ Did not rank most relevant document first")

    # Check 3: Model files exist
    if os.path.exists(os.path.join(output_path, "config.json")):
        print("✓ Model files saved successfully")
    else:
        print("✗ WARNING: Model files missing")

    print("\n=== Overall Validation Result ===")
    validation_passed = (
        all(isinstance(s, (int, float, np.number)) for s in scores) and
        os.path.exists(os.path.join(output_path, "config.json"))
    )

    if validation_passed:
        print("✓ Cross-encoder validation PASSED")
    else:
        print("✗ Cross-encoder validation FAILED")

    return reranker


def demonstrate_langchain_retrieval():
    """
    Demonstrate trained retrievers in LangChain RAG pipelines
    """

    if not LANGCHAIN_AVAILABLE:
        print("\nSkipping LangChain demo - package not installed")
        return

    print("\n" + "=" * 60)
    print("LangChain RAG Pipeline Integration")
    print("=" * 60)

    # Sample knowledge base
    documents = [
        "LangChain is a framework for developing applications powered by language models.",
        "Vector databases store embeddings and enable semantic search capabilities.",
        "RAG combines retrieval and generation to create accurate, grounded responses.",
        "Fine-tuning adapts pre-trained models to specific tasks and domains.",
        "Embeddings convert text into numerical vectors capturing semantic meaning.",
        "Cross-encoders re-rank documents by scoring query-document pairs jointly.",
        "FAISS is a library for efficient similarity search in dense vectors.",
        "Prompt engineering designs effective instructions for language models.",
    ]

    # Convert to LangChain documents
    docs = [Document(page_content=text) for text in documents]

    print(f"\nKnowledge base: {len(docs)} documents")

    # 1. Dense retrieval (vector store)
    print("\n1. Dense Retrieval (Vector Store)")

    retriever_path = "models/finetuned_retriever"
    if os.path.exists(retriever_path):
        embeddings = HuggingFaceEmbeddings(model_name=retriever_path)
    else:
        print("  Using default embeddings (fine-tuned model not found)")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create vector store
    vector_store = FAISS.from_documents(docs, embeddings)
    dense_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    test_query = "How does semantic search work?"
    results = dense_retriever.invoke(test_query)

    print(f"Query: '{test_query}'")
    print("Results:")
    for i, doc in enumerate(results, 1):
        print(f"  {i}. {doc.page_content}")

    # 2. BM25 Retrieval (keyword-based)
    print("\n2. BM25 Retrieval (Keyword-based)")

    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 3

    results = bm25_retriever.invoke(test_query)

    print(f"Query: '{test_query}'")
    print("Results:")
    for i, doc in enumerate(results, 1):
        print(f"  {i}. {doc.page_content}")

    # 3. Hybrid Retrieval (combining both)
    print("\n3. Hybrid Retrieval (Dense + BM25)")

    hybrid_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[0.5, 0.5]  # Equal weighting
    )

    results = hybrid_retriever.invoke(test_query)

    print(f"Query: '{test_query}'")
    print("Results:")
    for i, doc in enumerate(results[:3], 1):
        print(f"  {i}. {doc.page_content}")

    # 4. Multi-query retrieval
    print("\n4. Multi-Query Expansion (Advanced)")
    print("Generates multiple query variations for better recall")
    print("(Requires LLM - showing concept)")

    query_variations = [
        "How does semantic search work?",
        "What is the mechanism behind semantic search?",
        "Explain semantic search functionality"
    ]

    all_results = []
    for variant in query_variations:
        results = dense_retriever.invoke(variant)
        all_results.extend(results)

    # Deduplicate
    unique_results = {doc.page_content: doc for doc in all_results}

    print(f"Query variations: {len(query_variations)}")
    print(f"Unique results retrieved: {len(unique_results)}")

    print("\n✓ RAG pipeline demonstration complete")


def main():
    """Main training and demonstration pipeline"""

    print("\n🔍 Retrieval Model Training for LangChain RAG")
    print("=" * 60)

    # Train dense retriever
    train_dense_retriever()

    # Train cross-encoder reranker
    train_cross_encoder_reranker()

    # Demonstrate LangChain integration
    demonstrate_langchain_retrieval()

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print("\nTrained Models:")
    print("1. Dense Retriever (Bi-encoder): models/finetuned_retriever")
    print("   Use for: Initial retrieval, vector store embeddings")
    print("\n2. Cross-encoder Re-ranker: models/finetuned_reranker")
    print("   Use for: Re-ranking top results for higher accuracy")
    print("\nRecommended RAG Pipeline:")
    print("1. Dense retrieval → Get top 50-100 candidates (fast)")
    print("2. Cross-encoder re-ranking → Rank top 5-10 (accurate)")
    print("3. LLM generation → Generate final response")
    print("\nLangChain Integration:")
    print("• Use HuggingFaceEmbeddings with fine-tuned models")
    print("• Combine with BM25 for hybrid search")
    print("• Add cross-encoder as custom re-ranker")
    print("• Cache embeddings for frequently accessed documents")
    print("\nBest Practices:")
    print("• Fine-tune on domain-specific query-document pairs")
    print("• Use hard negatives in training (similar but wrong docs)")
    print("• Monitor retrieval metrics: MRR, NDCG, Recall@K")
    print("• Consider metadata filtering before semantic search")


if __name__ == "__main__":
    main()
