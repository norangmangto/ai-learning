"""
Sentiment Analysis with TF-IDF + LogisticRegression
Alternative to transformer-based approaches
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def train():
    print("Training Sentiment Analysis with TF-IDF + Logistic Regression...")

    try:
        # 1. Load Data
        dataset = load_dataset("imdb", split="train[:2000]")
        test_dataset = load_dataset("imdb", split="test[:500]")
    except:
        print("Warning: Could not load IMDB dataset. Using synthetic data...")
        dataset = create_synthetic_dataset(2000)
        test_dataset = create_synthetic_dataset(500)

    # 2. Extract texts and labels
    texts = [item['text'] for item in dataset]
    labels = np.array([item['label'] for item in dataset])

    test_texts = [item['text'] for item in test_dataset]
    test_labels = np.array([item['label'] for item in test_dataset])

    # 3. TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), max_df=0.8, min_df=2)
    X_train = vectorizer.fit_transform(texts)
    X_test = vectorizer.transform(test_texts)

    print(f"TF-IDF Feature Shape: {X_train.shape}")

    # 4. Train Model
    model = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
    model.fit(X_train, labels)

    print("✓ Model trained successfully")

    # 5. Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions)

    print(f"\nTF-IDF Sentiment Analysis Accuracy: {accuracy:.4f}")

    # 6. QA Validation
    print("\n=== QA Validation ===")
    cm = confusion_matrix(test_labels, predictions)
    print(f"F1-Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    print("\n--- Sanity Checks ---")
    if accuracy >= 0.7:
        print(f"✓ Good accuracy: {accuracy:.4f}")
    else:
        print(f"⚠ Moderate accuracy: {accuracy:.4f}")

    print("\n=== Overall Validation Result ===")
    validation_passed = accuracy >= 0.65 and f1 >= 0.6

    if validation_passed:
        print("✓ Validation PASSED")
    else:
        print("✗ Validation FAILED")

    # 7. Test on sample
    print("\n=== Sample Predictions ===")
    sample_texts = [
        "This movie is absolutely fantastic and amazing!",
        "Terrible waste of time, very boring movie.",
        "Not bad, could be better but decent enough."
    ]

    sample_X = vectorizer.transform(sample_texts)
    sample_preds = model.predict(sample_X)

    for text, pred in zip(sample_texts, sample_preds):
        sentiment = "POSITIVE" if pred == 1 else "NEGATIVE"
        print(f"Text: {text}")
        print(f"Prediction: {sentiment}\n")

    return model, vectorizer


def create_synthetic_dataset(size):
    """Create synthetic sentiment dataset"""
    positive_samples = [
        "This is absolutely amazing and wonderful! Simply the best!",
        "Excellent movie, I really loved it. Fantastic acting!",
        "Great experience, very good quality and service.",
        "Wonderful product, highly recommend!",
        "Excellent work, truly fantastic!",
    ]

    negative_samples = [
        "Terrible waste of time, absolutely horrible!",
        "This is the worst thing I've ever seen. Hate it!",
        "Awful quality, boring and disappointing.",
        "Bad experience, I regret buying this.",
        "Horrible movie, truly terrible!",
    ]

    data = []
    for _ in range(size // 10):
        for text in positive_samples:
            data.append({"text": text, "label": 1})
        for text in negative_samples:
            data.append({"text": text, "label": 0})

    return data[:size]


if __name__ == "__main__":
    train()
