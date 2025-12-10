"""
Doc2Vec (Paragraph Vectors) for Document Embeddings with Gensim

Doc2Vec extends Word2Vec to learn document-level representations:
- PV-DM (Distributed Memory): Like CBOW but includes document vector
- PV-DBOW (Distributed Bag of Words): Like Skip-gram for documents
- Captures semantic meaning of entire documents
- Useful for document similarity and classification

This implementation includes:
- Both PV-DM and PV-DBOW architectures
- Document similarity computation
- Document clustering
- Comparison with averaged Word2Vec
"""

import numpy as np
import matplotlib.pyplot as plt
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import Word2Vec
import time
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def generate_document_corpus(n_documents=500, doc_length_range=(20, 50)):
    """Generate synthetic document corpus with clear topics."""
    print(f"Generating {n_documents} documents...")

    np.random.seed(42)

    # Define topic-specific vocabularies
    topics = {
        'technology': {
            'core': ['computer', 'software', 'algorithm', 'data', 'network', 'programming',
                    'artificial', 'intelligence', 'machine', 'learning', 'neural', 'system',
                    'digital', 'technology', 'innovation', 'developer', 'code', 'application'],
            'related': ['design', 'create', 'build', 'develop', 'implement', 'process',
                       'analyze', 'optimize', 'integrate', 'deploy']
        },
        'nature': {
            'core': ['tree', 'forest', 'mountain', 'river', 'ocean', 'animal', 'bird',
                    'flower', 'plant', 'weather', 'climate', 'environment', 'natural',
                    'wildlife', 'ecosystem', 'habitat', 'species', 'conservation'],
            'related': ['beautiful', 'green', 'wild', 'natural', 'outdoor', 'scenic',
                       'preserve', 'protect', 'explore', 'discover']
        },
        'food': {
            'core': ['cook', 'recipe', 'ingredient', 'restaurant', 'meal', 'breakfast',
                    'dinner', 'lunch', 'delicious', 'flavor', 'taste', 'kitchen',
                    'chef', 'cuisine', 'dish', 'dessert', 'spice', 'fresh'],
            'related': ['prepare', 'serve', 'enjoy', 'savor', 'delicious', 'tasty',
                       'healthy', 'organic', 'gourmet', 'homemade']
        },
        'sports': {
            'core': ['game', 'play', 'team', 'player', 'score', 'win', 'competition',
                    'tournament', 'champion', 'athlete', 'training', 'exercise',
                    'match', 'victory', 'sport', 'fitness', 'league', 'season'],
            'related': ['compete', 'practice', 'improve', 'perform', 'achieve',
                       'challenge', 'succeed', 'excel', 'dominate', 'defeat']
        },
        'business': {
            'core': ['company', 'market', 'product', 'customer', 'sales', 'profit',
                    'strategy', 'management', 'finance', 'investment', 'revenue',
                    'growth', 'industry', 'enterprise', 'corporate', 'economy'],
            'related': ['increase', 'expand', 'develop', 'manage', 'optimize',
                       'improve', 'innovate', 'compete', 'succeed', 'achieve']
        }
    }

    documents = []
    labels = []
    topic_list = list(topics.keys())

    for i in range(n_documents):
        # Assign topic
        topic_idx = i % len(topic_list)
        topic = topic_list[topic_idx]
        topic_vocab = topics[topic]

        # Generate document
        doc_length = np.random.randint(*doc_length_range)
        doc = []

        for _ in range(doc_length):
            if np.random.rand() < 0.6:
                # Core topic words
                doc.append(np.random.choice(topic_vocab['core']))
            elif np.random.rand() < 0.8:
                # Related topic words
                doc.append(np.random.choice(topic_vocab['related']))
            else:
                # Common words
                common = ['the', 'a', 'an', 'is', 'are', 'was', 'were', 'and', 'or',
                         'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about']
                doc.append(np.random.choice(common))

        documents.append(doc)
        labels.append(topic)

    return documents, labels


def create_tagged_documents(documents):
    """Create TaggedDocument objects for Doc2Vec."""
    tagged_docs = []
    for i, doc in enumerate(documents):
        tagged_docs.append(TaggedDocument(words=doc, tags=[str(i)]))
    return tagged_docs


def train_doc2vec_dm(tagged_docs, vector_size=100, window=5, epochs=50):
    """Train Doc2Vec with PV-DM (Distributed Memory) model."""
    print("\nTraining Doc2Vec (PV-DM)...")
    print(f"Vector size: {vector_size}, Window: {window}, Epochs: {epochs}")

    model = Doc2Vec(
        documents=tagged_docs,
        vector_size=vector_size,
        window=window,
        min_count=2,
        workers=4,
        epochs=epochs,
        dm=1,  # PV-DM
        dm_mean=0,  # Use sum instead of mean
        alpha=0.025,
        min_alpha=0.0001
    )

    return model


def train_doc2vec_dbow(tagged_docs, vector_size=100, window=5, epochs=50):
    """Train Doc2Vec with PV-DBOW (Distributed Bag of Words) model."""
    print("\nTraining Doc2Vec (PV-DBOW)...")
    print(f"Vector size: {vector_size}, Window: {window}, Epochs: {epochs}")

    model = Doc2Vec(
        documents=tagged_docs,
        vector_size=vector_size,
        window=window,
        min_count=2,
        workers=4,
        epochs=epochs,
        dm=0,  # PV-DBOW
        alpha=0.025,
        min_alpha=0.0001
    )

    return model


def get_averaged_word2vec(doc, word2vec_model):
    """Get document vector by averaging Word2Vec vectors."""
    vectors = []
    for word in doc:
        if word in word2vec_model.wv:
            vectors.append(word2vec_model.wv[word])

    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)


def compare_document_embeddings(documents, labels):
    """Compare Doc2Vec with averaged Word2Vec."""
    print("\n" + "="*70)
    print("Comparing Document Embedding Methods")
    print("="*70)

    # Prepare data
    tagged_docs = create_tagged_documents(documents)

    # Train models
    start_time = time.time()
    dm_model = train_doc2vec_dm(tagged_docs, vector_size=100, window=5, epochs=30)
    dm_time = time.time() - start_time

    start_time = time.time()
    dbow_model = train_doc2vec_dbow(tagged_docs, vector_size=100, window=5, epochs=30)
    dbow_time = time.time() - start_time

    # Train Word2Vec for comparison
    start_time = time.time()
    word2vec_model = Word2Vec(sentences=documents, vector_size=100, window=5,
                              min_count=2, workers=4, epochs=30)
    w2v_time = time.time() - start_time

    print("\n" + "="*70)
    print("Training Times")
    print("="*70)
    print(f"PV-DM: {dm_time:.2f}s")
    print(f"PV-DBOW: {dbow_time:.2f}s")
    print(f"Word2Vec (for comparison): {w2v_time:.2f}s")

    # Get document vectors
    dm_vectors = np.array([dm_model.dv[str(i)] for i in range(len(documents))])
    dbow_vectors = np.array([dbow_model.dv[str(i)] for i in range(len(documents))])
    w2v_vectors = np.array([get_averaged_word2vec(doc, word2vec_model) for doc in documents])

    # Evaluate with clustering
    print("\n" + "="*70)
    print("Clustering Quality (Silhouette Score)")
    print("="*70)

    n_clusters = len(set(labels))

    for name, vectors in [('PV-DM', dm_vectors), ('PV-DBOW', dbow_vectors),
                          ('Averaged Word2Vec', w2v_vectors)]:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        pred_labels = kmeans.fit_predict(vectors)
        score = silhouette_score(vectors, pred_labels)
        print(f"{name:20} : {score:.4f}")

    return dm_model, dbow_model, word2vec_model


def document_similarity(model, doc_id1, doc_id2, documents):
    """Calculate similarity between two documents."""
    vec1 = model.dv[str(doc_id1)]
    vec2 = model.dv[str(doc_id2)]
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    print(f"\nDocument {doc_id1}: {' '.join(documents[doc_id1][:15])}...")
    print(f"Document {doc_id2}: {' '.join(documents[doc_id2][:15])}...")
    print(f"Similarity: {similarity:.4f}")

    return similarity


def find_similar_documents(model, doc_id, topn=5):
    """Find most similar documents."""
    similar = model.dv.most_similar(str(doc_id), topn=topn)

    print(f"\nMost similar documents to document {doc_id}:")
    for similar_id, score in similar:
        print(f"  Document {similar_id:10} : {score:.4f}")

    return similar


def visualize_document_embeddings(model, labels, n_docs=200, method='tsne'):
    """Visualize document embeddings."""
    print(f"\nVisualizing {n_docs} documents using {method.upper()}...")

    # Get document vectors
    doc_vectors = np.array([model.dv[str(i)] for i in range(min(n_docs, len(model.dv)))])
    doc_labels = labels[:min(n_docs, len(labels))]

    # Reduce to 2D
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    else:  # PCA
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)

    vectors_2d = reducer.fit_transform(doc_vectors)

    # Plot
    plt.figure(figsize=(12, 8))

    # Get unique labels and assign colors
    unique_labels = list(set(doc_labels))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}

    for label in unique_labels:
        mask = [l == label for l in doc_labels]
        plt.scatter(vectors_2d[mask, 0], vectors_2d[mask, 1],
                   c=[label_to_color[label]], label=label, s=50, alpha=0.6)

    plt.title(f'Document Embeddings Visualization ({method.upper()})', fontsize=14)
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'doc2vec_embeddings_{method}.png', dpi=300, bbox_inches='tight')
    plt.show()


def infer_new_document(model, new_doc):
    """Infer vector for a new document."""
    print(f"\nInferring vector for new document: {' '.join(new_doc[:20])}...")

    # Infer vector
    vector = model.infer_vector(new_doc)

    # Find similar documents
    similarities = []
    for i in range(len(model.dv)):
        doc_vec = model.dv[str(i)]
        sim = np.dot(vector, doc_vec) / (np.linalg.norm(vector) * np.linalg.norm(doc_vec))
        similarities.append((i, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)

    print("Most similar training documents:")
    for doc_id, score in similarities[:5]:
        print(f"  Document {doc_id:5} : {score:.4f}")

    return vector


def evaluate_on_classification_task(model, documents, labels, test_size=0.2):
    """Evaluate document embeddings on classification task."""
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report

    print("\n" + "="*70)
    print("Document Classification Evaluation")
    print("="*70)

    # Get document vectors
    X = np.array([model.dv[str(i)] for i in range(len(documents))])
    y = labels

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Train classifier
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nClassification Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


def main():
    """Main execution function."""
    print("="*70)
    print("Doc2Vec Document Embeddings")
    print("="*70)

    # Generate corpus
    print("\n1. Generating document corpus...")
    documents, labels = generate_document_corpus(n_documents=600, doc_length_range=(20, 50))
    print(f"Generated {len(documents)} documents across {len(set(labels))} topics")
    print(f"Topics: {', '.join(set(labels))}")

    # Compare embedding methods
    print("\n2. Comparing embedding methods...")
    dm_model, dbow_model, w2v_model = compare_document_embeddings(documents, labels)

    # Test document similarity
    print("\n3. Testing document similarity...")
    document_similarity(dm_model, 0, 1, documents)
    document_similarity(dm_model, 0, 5, documents)

    # Find similar documents
    print("\n4. Finding similar documents...")
    find_similar_documents(dm_model, 0, topn=5)

    # Visualize embeddings
    print("\n5. Visualizing document embeddings...")
    visualize_document_embeddings(dm_model, labels, n_docs=300, method='tsne')

    # Infer vector for new document
    print("\n6. Inferring vector for new document...")
    new_doc = ['computer', 'algorithm', 'software', 'programming', 'neural', 'network']
    infer_new_document(dm_model, new_doc)

    # Evaluate on classification
    print("\n7. Evaluating on classification task...")
    evaluate_on_classification_task(dm_model, documents, labels, test_size=0.2)

    # Save models
    dm_model.save('doc2vec_dm_model.bin')
    dbow_model.save('doc2vec_dbow_model.bin')
    print("\nModels saved to 'doc2vec_dm_model.bin' and 'doc2vec_dbow_model.bin'")

    print("\n" + "="*70)
    print("Doc2Vec Training Complete!")
    print("="*70)
    print("\nKey Features:")
    print("✓ PV-DM: Includes document vector in word prediction (like CBOW)")
    print("✓ PV-DBOW: Predicts words from document vector (like Skip-gram)")
    print("✓ Learns fixed-length representations for variable-length documents")
    print("✓ Can infer vectors for new, unseen documents")
    print("\nDoc2Vec vs Averaged Word2Vec:")
    print("  • Doc2Vec: Learns document-specific context, better semantic understanding")
    print("  • Avg Word2Vec: Simple averaging, loses word order information")
    print("\nBest for: Document classification, clustering, similarity, recommendation")


if __name__ == "__main__":
    main()
