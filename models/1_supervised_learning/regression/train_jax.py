import jax
import jax.numpy as jnp
from jax import grad, vmap
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def train():
    print("Training Linear Regression with JAX...")

    # 1. Prepare Data
    X, y = make_regression(n_samples=1000, n_features=1, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = jnp.array(X_train, dtype=jnp.float32)
    y_train = jnp.array(y_train, dtype=jnp.float32)
    X_test = jnp.array(X_test, dtype=jnp.float32)
    y_test = jnp.array(y_test, dtype=jnp.float32)

    # 2. Initialize parameters
    key = jax.random.PRNGKey(42)
    w = jax.random.normal(key, shape=(1, 1)) * 0.01
    b = jnp.array([0.0])

    # 3. Define loss function
    def predict(params, x):
        w, b = params
        return jnp.dot(x, w) + b

    def loss_fn(params, x, y):
        predictions = vmap(lambda xi: predict(params, xi))(x)
        return jnp.mean((predictions - y) ** 2)

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
    predictions = vmap(lambda xi: predict(params, xi))(X_test)
    mse = np.mean((np.array(predictions) - y_test) ** 2)

    print(f"JAX Linear Regression MSE: {mse:.4f}")

    # 6. QA Validation
    print("\n=== QA Validation ===")
    mae = np.mean(np.abs(np.array(predictions) - y_test))
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mse)

    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")

    print("\n--- Sanity Checks ---")
    if np.all(np.isfinite(predictions)):
        print("✓ All predictions are finite")
    else:
        print("✗ WARNING: Some predictions are NaN or Inf!")

    if r2 > 0.5:
        print(f"✓ Good R² score: {r2:.4f}")
    elif r2 > 0:
        print(f"⚠ Moderate R² score: {r2:.4f}")
    else:
        print(f"✗ WARNING: Poor R² score: {r2:.4f}")

    print("\n=== Overall Validation Result ===")
    validation_passed = np.all(np.isfinite(predictions)) and r2 > 0 and mse < np.var(y_test) * 2

    if validation_passed:
        print("✓ Validation PASSED")
    else:
        print("✗ Validation FAILED")

    return params


if __name__ == "__main__":
    train()
