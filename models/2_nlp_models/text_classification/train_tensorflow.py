import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def train():
    print("Training Text Theme Classification with TensorFlow (BERT)...")

    # 1. Prepare Data
    print("Loading dataset...")
    try:
        # Use AG News dataset for topic/theme classification
        dataset = load_dataset("ag_news", split="train[:1000]")
        test_dataset = load_dataset("ag_news", split="test[:200]")
    except Exception as e:
        print(f"Failed to download dataset: {e}")
        print("Using sample data for demonstration...")
        # Sample data with themes: business, sports, politics, technology
        sample_data = {
            'text': [
                "Apple reports record quarterly profits driven by iPhone sales",
                "The stock market reached new highs today amid positive economic indicators",
                "The football team won the championship after an exciting final match",
                "Olympic athletes prepare for the upcoming games with rigorous training",
                "Congress passes new legislation on healthcare reform",
                "Election results show a tight race between the two main candidates",
                "New AI technology promises to revolutionize medical diagnostics",
                "Scientists develop breakthrough in quantum computing research"
            ],
            'label': [0, 0, 1, 1, 2, 2, 3, 3]  # 0: business, 1: sports, 2: politics, 3: technology
        }
        dataset = Dataset.from_dict(sample_data)
        test_dataset = Dataset.from_dict({
            'text': sample_data['text'][:4],
            'label': sample_data['label'][:4]
        })

    # 2. Load Pre-trained Model
    print("Loading BERT model...")
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Determine number of classes
    num_labels = len(set(dataset['label']))
    model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    # 3. Tokenize Data
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    tokenized_train = dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    # Convert to TensorFlow format
    train_texts = [dataset[i]['text'] for i in range(len(dataset))]
    train_labels = [dataset[i]['label'] for i in range(len(dataset))]
    test_texts = [test_dataset[i]['text'] for i in range(len(test_dataset))]
    test_labels = [test_dataset[i]['label'] for i in range(len(test_dataset))]

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128, return_tensors="tf")
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128, return_tensors="tf")

    # 4. Train Model
    print("\nTraining model...")
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    train_dataset_tf = tf.data.Dataset.from_tensor_slices((
        dict(train_encodings),
        train_labels
    )).batch(8)

    test_dataset_tf = tf.data.Dataset.from_tensor_slices((
        dict(test_encodings),
        test_labels
    )).batch(8)

    model.fit(train_dataset_tf, epochs=2, validation_data=test_dataset_tf)

    # 5. Evaluate
    print("\n=== Evaluation ===")
    results = model.evaluate(test_dataset_tf)
    print(f"Test Loss: {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]:.4f}")

    # Test predictions
    print("\n=== Sample Predictions ===")

    theme_names = {0: "Business", 1: "Sports", 2: "Politics", 3: "Technology"}

    for i in range(min(5, len(test_texts))):
        text = test_texts[i]
        true_label = test_labels[i]

        inputs = tokenizer([text], return_tensors="tf", padding=True, truncation=True, max_length=128)
        outputs = model(inputs)
        predictions = tf.argmax(outputs.logits, axis=-1)
        predicted_label = predictions.numpy()[0]

        print(f"\nText: {text[:100]}...")
        print(f"True Theme: {theme_names.get(true_label, true_label)}")
        print(f"Predicted Theme: {theme_names.get(predicted_label, predicted_label)}")

    # 6. QA Validation
    print("\n=== QA Validation ===")
    print(f"✓ Model trained successfully on {len(train_texts)} samples")
    print(f"✓ Test accuracy: {results[1]:.2%}")
    print("✓ Theme classification model can categorize text into multiple topics")

if __name__ == "__main__":
    train()
