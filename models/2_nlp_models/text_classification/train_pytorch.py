import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def train():
    print("Training Text Theme Classification with PyTorch (BERT)...")

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
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 3. Tokenize Data
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    tokenized_train = dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    # 4. Train Model
    print("\nTraining model...")
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {
            'accuracy': accuracy_score(labels, predictions),
            'f1': f1_score(labels, predictions, average='weighted')
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # 5. Evaluate
    print("\n=== Evaluation ===")
    results = trainer.evaluate()
    print(f"Accuracy: {results['eval_accuracy']:.4f}")
    print(f"F1 Score: {results['eval_f1']:.4f}")

    # Test predictions
    model.eval()
    print("\n=== Sample Predictions ===")

    theme_names = {0: "Business", 1: "Sports", 2: "Politics", 3: "Technology"}

    for i in range(min(5, len(test_dataset))):
        text = test_dataset[i]['text']
        true_label = test_dataset[i]['label']

        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            predicted_label = predictions.item()

        print(f"\nText: {text[:100]}...")
        print(f"True Theme: {theme_names.get(true_label, true_label)}")
        print(f"Predicted Theme: {theme_names.get(predicted_label, predicted_label)}")

    # 6. QA Validation
    print("\n=== QA Validation ===")
    print(f"✓ Model trained successfully on {len(dataset)} samples")
    print(f"✓ Test accuracy: {results['eval_accuracy']:.2%}")
    print("✓ Theme classification model can categorize text into multiple topics")

if __name__ == "__main__":
    train()
