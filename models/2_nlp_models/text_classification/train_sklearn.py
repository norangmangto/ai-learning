"""
Text Classification with FastText
Alternative to transformer-based approaches (lightweight and fast)
"""

import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
import tempfile
import os


def train():
    print("Training Text Classification with FastText...")

    try:
        # FastText is a separate library, attempt installation if not available
        import fasttext
    except ImportError:
        print(
            "Warning: fasttext package not installed. Simulating with simple classifier..."
        )
        return train_simple_classifier()

    try:
        # 1. Load Data
        dataset = load_dataset("ag_news", split="train[:5000]")
        test_dataset = load_dataset("ag_news", split="test[:1000]")
    except:
        print("Warning: Could not load AG_News dataset. Using synthetic data...")
        dataset = create_synthetic_dataset(5000)
        test_dataset = create_synthetic_dataset(1000)

    # 2. Prepare data in FastText format
    with tempfile.TemporaryDirectory() as tmpdir:
        train_file = os.path.join(tmpdir, "train.txt")
        test_file = os.path.join(tmpdir, "test.txt")

        # Write training data
        with open(train_file, "w") as f:
            for item in dataset:
                label = f"__label__{item['label']}"
                text = item["text"].replace("\n", " ")
                f.write(f"{label} {text}\n")

        # Write test data
        with open(test_file, "w") as f:
            for item in test_dataset:
                label = f"__label__{item['label']}"
                text = item["text"].replace("\n", " ")
                f.write(f"{label} {text}\n")

        # 3. Train FastText Model
        model = fasttext.train_supervised(
            input=train_file, epoch=10, lr=0.5, wordNgrams=2
        )

        print("✓ FastText model trained successfully")

        # 4. Evaluate
        test_texts = [item["text"].replace("\n", " ") for item in test_dataset]
        test_labels = np.array([item["label"] for item in test_dataset])

        predictions = []
        for text in test_texts:
            pred = model.predict(text)
            predictions.append(int(pred[0][0].split("__")[-1]))
        predictions = np.array(predictions)

        accuracy = accuracy_score(test_labels, predictions)
        print(f"\nFastText Classification Accuracy: {accuracy:.4f}")

        # 5. QA Validation
        print("\n=== QA Validation ===")
        f1 = f1_score(test_labels, predictions, average="weighted")
        print(f"F1-Score (weighted): {f1:.4f}")

        print("\n--- Sanity Checks ---")
        if accuracy >= 0.7:
            print(f"✓ Good accuracy: {accuracy:.4f}")
        else:
            print(f"⚠ Moderate accuracy: {accuracy:.4f}")

        print("\n=== Overall Validation Result ===")
        validation_passed = accuracy >= 0.65

        if validation_passed:
            print("✓ Validation PASSED")
        else:
            print("✗ Validation FAILED")

        return model


def train_simple_classifier():
    """Fallback simple classifier when FastText is not available"""
    print("Using TF-IDF + Logistic Regression as fallback...")

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    dataset = create_synthetic_dataset(2000)
    test_dataset = create_synthetic_dataset(500)

    texts = [item["text"] for item in dataset]
    labels = np.array([item["label"] for item in dataset])

    test_texts = [item["text"] for item in test_dataset]
    test_labels = np.array([item["label"] for item in test_dataset])

    vectorizer = TfidfVectorizer(max_features=3000)
    X_train = vectorizer.fit_transform(texts)
    X_test = vectorizer.transform(test_texts)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, labels)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(test_labels, predictions)

    print(f"Fallback TF-IDF Classification Accuracy: {accuracy:.4f}")

    return model


def create_synthetic_dataset(size):
    """Create synthetic text classification dataset"""
    categories = {
        0: [  # Tech
            "New AI breakthrough announced today",
            "Latest smartphone features revealed",
            "Cloud computing advances continue",
        ],
        1: [  # Sports
            "Team wins championship game",
            "Athlete breaks world record",
            "Olympics preparation underway",
        ],
        2: [  # Business
            "Market reaches new high",
            "Company announces earnings",
            "Economic growth reported",
        ],
        3: [  # Entertainment
            "Movie wins major award",
            "Concert tickets sell out",
            "Celebrity announces new project",
        ],
    }

    data = []
    for _ in range(size // 12):
        for label, texts in categories.items():
            for text in texts:
                data.append({"text": text, "label": label})

    return data[:size]


if __name__ == "__main__":
    train()
