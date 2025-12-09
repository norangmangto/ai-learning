"""
Training and Fine-tuning Embedding Models for LangChain
This script demonstrates how to work with embedding models commonly used in LangChain:
- OpenAI Embeddings
- HuggingFace Embeddings
- Sentence Transformers
- Custom fine-tuning for domain-specific embeddings
"""

import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from sentence_transformers.util import cos_sim
from torch.utils.data import DataLoader
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
import os

# For LangChain integration
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_openai import OpenAIEmbeddings
    LANGCHAIN_AVAILABLE = True
except ImportError:
    print("⚠ LangChain not installed. Install with: pip install langchain langchain-community langchain-openai")
    LANGCHAIN_AVAILABLE = False


def create_training_data():
    """
    Create sample training data for embedding fine-tuning.
    In practice, use your domain-specific data.
    """
    # Format: (text1, text2, similarity_score)
    # Score: 0 = dissimilar, 1 = similar
    train_examples = [
        ("What is machine learning?", "Define machine learning", 1.0),
        ("What is machine learning?", "How to cook pasta", 0.0),
        ("Python programming tutorial", "Learn Python coding", 0.9),
        ("Python programming tutorial", "Java enterprise development", 0.3),
        ("Natural language processing", "NLP techniques", 1.0),
        ("Natural language processing", "Computer vision methods", 0.4),
        ("Deep learning neural networks", "Artificial neural networks explained", 0.95),
        ("Deep learning neural networks", "Traditional statistics methods", 0.2),
        ("Data science best practices", "Data analysis techniques", 0.85),
        ("Data science best practices", "Mobile app development", 0.1),
        ("RAG architecture patterns", "Retrieval augmented generation", 1.0),
        ("RAG architecture patterns", "Database normalization", 0.15),
        ("LangChain framework usage", "Using LangChain for AI apps", 0.95),
        ("LangChain framework usage", "React frontend framework", 0.0),
        ("Vector database indexing", "Semantic search with vectors", 0.8),
        ("Vector database indexing", "SQL query optimization", 0.3),
    ]
    
    # Create InputExamples for training
    train_data = []
    for text1, text2, score in train_examples:
        train_data.append(InputExample(texts=[text1, text2], label=float(score)))
    
    # Create evaluation data
    eval_examples = [
        ("Explain transformers", "Transformer architecture", 1.0),
        ("Explain transformers", "Electric transformers", 0.1),
        ("Fine-tuning LLMs", "LLM adaptation techniques", 0.9),
        ("Fine-tuning LLMs", "Web scraping tools", 0.0),
    ]
    
    eval_data = []
    for text1, text2, score in eval_examples:
        eval_data.append(InputExample(texts=[text1, text2], label=float(score)))
    
    return train_data, eval_data


def train_sentence_transformer():
    """
    Fine-tune a Sentence Transformer model for domain-specific embeddings.
    This model can be used with LangChain's HuggingFaceEmbeddings.
    """
    print("=" * 60)
    print("Fine-tuning Sentence Transformer for LangChain")
    print("=" * 60)
    
    # 1. Load pre-trained model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Fast and efficient
    print(f"\nLoading base model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print(f"✓ Model loaded")
    print(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")
    
    # 2. Prepare training data
    print("\nPreparing training data...")
    train_data, eval_data = create_training_data()
    print(f"✓ Training examples: {len(train_data)}")
    print(f"✓ Evaluation examples: {len(eval_data)}")
    
    # 3. Create DataLoader
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=4)
    
    # 4. Define loss function
    # CosineSimilarityLoss is good for semantic similarity tasks
    train_loss = losses.CosineSimilarityLoss(model)
    
    # 5. Define evaluator
    evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(
        eval_data, 
        name='eval'
    )
    
    # 6. Fine-tune the model
    print("\nFine-tuning model...")
    output_path = "models/finetuned_embeddings"
    os.makedirs(output_path, exist_ok=True)
    
    epochs = 4
    warmup_steps = int(len(train_dataloader) * epochs * 0.1)
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_path,
        evaluation_steps=10,
        show_progress_bar=True
    )
    
    print(f"\n✓ Model fine-tuned and saved to: {output_path}")
    
    # 7. QA Validation
    print("\n" + "=" * 60)
    print("QA Validation")
    print("=" * 60)
    
    # Load the fine-tuned model
    finetuned_model = SentenceTransformer(output_path)
    
    # Test queries
    test_queries = [
        "What is machine learning?",
        "Explain deep learning",
        "How to use LangChain?"
    ]
    
    test_docs = [
        "Machine learning is a subset of AI that enables systems to learn from data",
        "Deep learning uses neural networks with multiple layers to learn representations",
        "LangChain is a framework for developing applications powered by language models",
        "Cooking recipes for Italian cuisine",
        "History of the Roman Empire"
    ]
    
    print("\n--- Semantic Similarity Test ---")
    query_embeddings = finetuned_model.encode(test_queries)
    doc_embeddings = finetuned_model.encode(test_docs)
    
    similarities = cos_sim(query_embeddings, doc_embeddings)
    
    for i, query in enumerate(test_queries):
        print(f"\nQuery: '{query}'")
        query_sims = similarities[i].numpy()
        top_idx = np.argsort(query_sims)[::-1][:3]
        
        for rank, idx in enumerate(top_idx, 1):
            print(f"  {rank}. (Score: {query_sims[idx]:.4f}) {test_docs[idx][:60]}...")
    
    # Sanity checks
    print("\n--- Sanity Checks ---")
    
    # Check 1: Embedding dimensions are consistent
    if all(emb.shape[0] == model.get_sentence_embedding_dimension() for emb in query_embeddings):
        print(f"✓ All embeddings have correct dimension: {model.get_sentence_embedding_dimension()}")
    else:
        print("✗ WARNING: Inconsistent embedding dimensions!")
    
    # Check 2: Embeddings are normalized (for cosine similarity)
    norms = np.linalg.norm(query_embeddings, axis=1)
    if np.allclose(norms, 1.0, atol=0.01):
        print("✓ Embeddings are normalized (good for cosine similarity)")
    else:
        print(f"⚠ Embeddings not normalized. Norms: {norms}")
    
    # Check 3: Similar queries have high similarity
    sim_ml_dl = cos_sim(query_embeddings[0:1], query_embeddings[1:2]).item()
    if sim_ml_dl > 0.6:
        print(f"✓ Related queries have high similarity: {sim_ml_dl:.4f}")
    else:
        print(f"⚠ Related queries have low similarity: {sim_ml_dl:.4f}")
    
    # Check 4: Model file exists
    model_file = os.path.join(output_path, "pytorch_model.bin")
    if os.path.exists(model_file):
        print(f"✓ Model file saved successfully")
    else:
        print(f"✗ WARNING: Model file not found")
    
    print("\n=== Overall Validation Result ===")
    validation_passed = (
        all(emb.shape[0] == model.get_sentence_embedding_dimension() for emb in query_embeddings) and
        os.path.exists(model_file) and
        sim_ml_dl > 0.5
    )
    
    if validation_passed:
        print("✓ Model validation PASSED - Ready for LangChain integration")
    else:
        print("✗ Model validation FAILED - Review training process")
    
    return finetuned_model


def test_langchain_embeddings():
    """
    Test various embedding models available in LangChain
    """
    if not LANGCHAIN_AVAILABLE:
        print("Skipping LangChain tests - package not installed")
        return
    
    print("\n" + "=" * 60)
    print("Testing LangChain Embedding Models")
    print("=" * 60)
    
    test_texts = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "Explain natural language processing"
    ]
    
    # 1. HuggingFace Embeddings (local, free)
    print("\n1. HuggingFace Embeddings")
    try:
        hf_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        embeddings = hf_embeddings.embed_documents(test_texts)
        query_embedding = hf_embeddings.embed_query(test_texts[0])
        
        print(f"✓ Model: all-MiniLM-L6-v2")
        print(f"✓ Embedding dimension: {len(embeddings[0])}")
        print(f"✓ Document embeddings: {len(embeddings)}")
        print(f"✓ Query embedding shape: {len(query_embedding)}")
        
        # Calculate similarity
        query_emb = np.array(query_embedding).reshape(1, -1)
        doc_embs = np.array(embeddings)
        similarities = cosine_similarity(query_emb, doc_embs)[0]
        
        print(f"\nSimilarities to query '{test_texts[0]}':")
        for i, (text, sim) in enumerate(zip(test_texts, similarities)):
            print(f"  {i+1}. {sim:.4f} - {text}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # 2. OpenAI Embeddings (requires API key)
    print("\n2. OpenAI Embeddings")
    if os.getenv("OPENAI_API_KEY"):
        try:
            openai_embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small"  # Latest, efficient model
            )
            
            embeddings = openai_embeddings.embed_documents(test_texts[:2])  # Limit to save costs
            
            print(f"✓ Model: text-embedding-3-small")
            print(f"✓ Embedding dimension: {len(embeddings[0])}")
            print(f"✓ Note: This uses OpenAI API (costs apply)")
            
        except Exception as e:
            print(f"✗ Error: {e}")
    else:
        print("⚠ Skipped - Set OPENAI_API_KEY environment variable to test")
    
    # 3. Custom fine-tuned model
    print("\n3. Custom Fine-tuned Model")
    output_path = "models/finetuned_embeddings"
    if os.path.exists(output_path):
        try:
            custom_embeddings = HuggingFaceEmbeddings(
                model_name=output_path
            )
            
            embeddings = custom_embeddings.embed_documents(test_texts)
            
            print(f"✓ Loaded custom fine-tuned model")
            print(f"✓ Embedding dimension: {len(embeddings[0])}")
            print(f"✓ Ready for domain-specific tasks")
            
        except Exception as e:
            print(f"✗ Error: {e}")
    else:
        print("⚠ Fine-tuned model not found - run training first")


def demonstrate_rag_pipeline():
    """
    Demonstrate how to use trained embeddings in a LangChain RAG pipeline
    """
    if not LANGCHAIN_AVAILABLE:
        return
    
    print("\n" + "=" * 60)
    print("RAG Pipeline Integration Example")
    print("=" * 60)
    
    from langchain_community.vectorstores import FAISS
    from langchain.text_splitter import CharacterTextSplitter
    
    # Sample documents
    documents = [
        "LangChain is a framework for developing applications powered by language models.",
        "Vector databases store embeddings for semantic search and retrieval.",
        "RAG (Retrieval Augmented Generation) combines retrieval with language model generation.",
        "Fine-tuning embeddings improves retrieval accuracy for domain-specific tasks.",
        "Sentence transformers create dense vector representations of text.",
    ]
    
    print("\nSample Knowledge Base:")
    for i, doc in enumerate(documents, 1):
        print(f"  {i}. {doc}")
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Create vector store
    print("\nBuilding vector store...")
    vector_store = FAISS.from_texts(documents, embeddings)
    print("✓ Vector store created")
    
    # Test retrieval
    queries = [
        "What is RAG?",
        "How do vector databases work?",
        "Tell me about LangChain"
    ]
    
    print("\n--- Retrieval Tests ---")
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = vector_store.similarity_search(query, k=2)
        
        for i, doc in enumerate(results, 1):
            print(f"  {i}. {doc.page_content}")
    
    print("\n✓ RAG pipeline demonstration complete")
    print("  This shows how embeddings integrate with LangChain's vector stores")


def main():
    """Main training and evaluation pipeline"""
    
    print("\n🔤 Embedding Model Training for LangChain")
    print("=" * 60)
    
    # Train custom embeddings
    model = train_sentence_transformer()
    
    # Test LangChain integrations
    test_langchain_embeddings()
    
    # Demonstrate RAG pipeline
    demonstrate_rag_pipeline()
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print("\nNext Steps:")
    print("1. Use fine-tuned model: models/finetuned_embeddings")
    print("2. Integrate with LangChain:")
    print("   embeddings = HuggingFaceEmbeddings(model_name='models/finetuned_embeddings')")
    print("3. Build RAG applications with domain-specific embeddings")
    print("4. Consider training on larger domain-specific datasets")
    print("\nRecommended LangChain Models:")
    print("• all-MiniLM-L6-v2: Fast, 384 dim (best for most use cases)")
    print("• all-mpnet-base-v2: High quality, 768 dim (better accuracy)")
    print("• text-embedding-3-small: OpenAI, 1536 dim (requires API key)")
    print("• text-embedding-3-large: OpenAI, 3072 dim (best quality, costly)")


if __name__ == "__main__":
    main()
