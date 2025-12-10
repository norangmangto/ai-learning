import jax
import jax.numpy as jnp
from jax import grad, vmap
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


def train():
    print("Training Logistic Regression with JAX...")

    # 1. Prepare Data
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Normalize
    X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
    X_train = (X_train - X_mean) / (X_std + 1e-8)
    X_test = (X_test - X_mean) / (X_std + 1e-8)

    X_train = jnp.array(X_train, dtype=jnp.float32)
    y_train = jnp.array(y_train, dtype=jnp.float32)
    X_test = jnp.array(X_test, dtype=jnp.float32)
    y_test = jnp.array(y_test, dtype=jnp.float32)

    # 2. Initialize parameters
    key = jax.random.PRNGKey(42)
    w = jax.random.normal(key, shape=(20, 1)) * 0.01
    b = jnp.array([0.0])

    # 3. Define loss function
    def sigmoid(x):
        return 1.0 / (1.0 + jnp.exp(-x))

    def predict(params, x):
        w, b = params
        return sigmoid(jnp.dot(x, w) + b)

    def loss_fn(params, x, y):
        predictions = vmap(lambda xi, yi: predict(params, xi))(x, y)
        epsilon = 1e-7
        predictions = jnp.clip(predictions, epsilon, 1 - epsilon)
        return -jnp.mean(y * jnp.log(predictions) + (1 - y) * jnp.log(1 - predictions))

    # 4. Gradient descent
    grad_fn = grad(loss_fn)
    learning_rate = 0.01
    epochs = 100
    params = (w, b)

    for epoch in range(epochs):
        grads = grad_fn(params, X_train, y_train)
        w_grad, b_grad = grads
        w = w - learning_rate * w_grad
        b = b - learning_rate * b_grad
        params = (w, b)

        if (epoch + 1) % 10 == 0:
            loss = loss_fn(params, X_train, y_train)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss:.4f}")

    # 5. Evaluate
    predictions = vmap(lambda xi: (predict(params, xi) > 0.5).astype(jnp.float32))(
        X_test
    )
    accuracy = accuracy_score(y_test, predictions)

    print(f"JAX Logistic Regression Accuracy: {accuracy:.4f}")

    # 6. QA Validation
    print("\n=== QA Validation ===")
    f1 = f1_score(y_test, predictions, average="binary")
    print(f"F1-Score: {f1:.4f}")

    print("\n--- Sanity Checks ---")
    if accuracy >= 0.5:
        print(f"✓ Good accuracy: {accuracy:.4f}")
    else:
        print(f"✗ WARNING: Poor accuracy: {accuracy:.4f}")

    print("\n=== Overall Validation Result ===")
    validation_passed = accuracy >= 0.6

    if validation_passed:
        print("✓ Validation PASSED")
    else:
        print("✗ Validation FAILED")

    return params


if __name__ == "__main__":
    train()
