"""
Linear Regression from Scratch (Gradient Descent)

Goals:
- Implement linear regression without sklearn
- Train using gradient descent
- Visualize loss over time
- Keep code readable and reproducible

Run:
  python linear_regression_gd.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)


def make_synthetic_data(n: int = 200, noise_std: float = 8.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic 1D linear data: y = 3x + 10 + noise
    Returns:
      X: shape (n, 1)
      y: shape (n, 1)
    """
    X = np.random.uniform(0, 50, size=(n, 1))
    true_w = 3.0
    true_b = 10.0
    noise = np.random.normal(0, noise_std, size=(n, 1))
    y = true_w * X + true_b + noise
    return X, y


def add_bias_column(X: np.ndarray) -> np.ndarray:
    """Convert X shape (n, d) -> Xb shape (n, d+1) with leading 1s for bias."""
    ones = np.ones((X.shape[0], 1))
    return np.hstack([ones, X])


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))

def normal_equation(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """ Closed-form solution for linear regression:
    theta = (X^T X)^(-1) X^T y """
    Xb = add_bias_column(X)
    return np.linalg.inv(Xb.T @ Xb) @ Xb.T @ y

def train_linear_regression_gd(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 1e-4,
    epochs: int = 2000,
) -> tuple[np.ndarray, list[float]]:
    """
    Train linear regression using gradient descent.

    Uses the model:
      y_hat = b + w * x   (for 1D X)
    but implemented generically via bias column.

    Returns:
      theta: shape (d+1, 1)  (theta[0] is bias)
      losses: list of MSE values
    """
    Xb = add_bias_column(X)              # (n, d+1)
    n, d1 = Xb.shape
    theta = np.zeros((d1, 1))            # (d+1, 1)

    losses: list[float] = []

    for epoch in range(epochs):
        y_pred = Xb @ theta              # (n, 1)
        error = y_pred - y               # (n, 1)

        # Gradient of MSE wrt theta:
        # d/dtheta (1/n * sum(error^2)) = (2/n) * Xb^T * error
        grad = (2.0 / n) * (Xb.T @ error)

        theta -= lr * grad

        if epoch % 50 == 0 or epoch == epochs - 1:
            loss = mean_squared_error(y, y_pred)
            losses.append(loss)

    return theta, losses


def predict(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    Xb = add_bias_column(X)
    return Xb @ theta


def main() -> None:
    set_seed(42)

    # 1) Data
    X, y = make_synthetic_data(n=250, noise_std=10.0)

    # 2) Train
    theta, losses = train_linear_regression_gd(X, y, lr=1e-4, epochs=5000)

    b = float(theta[0, 0])
    w = float(theta[1, 0])

    # Closed-form solution
    theta_ne = normal_equation(X, y)

    y_hat_ne = predict(X, theta_ne)
    mse_ne = mean_squared_error(y, y_hat_ne)

    print("=== Closed-form Solution (Normal Equation)===")
    print(f"Learned parameters: w={theta_ne[1,0]:.4f}, b={theta_ne[0,0]:.4f}")
    print(f"Final MSE: {mse_ne:.4f}")
    

    # 3) Report
    y_hat = predict(X, theta)
    mse = mean_squared_error(y, y_hat)

    print("=== Linear Regression (Gradient Descent) ===")
    print(f"Learned parameters: w={w:.4f}, b={b:.4f}")
    print(f"Final MSE: {mse:.4f}")

    # 4) Plot predictions
    order = np.argsort(X[:, 0])
    X_sorted = X[order]
    y_sorted = y[order]
    y_hat_sorted = y_hat[order]

    plt.figure()
    plt.scatter(X_sorted, y_sorted, s=10)
    plt.plot(X_sorted, y_hat_sorted)
    plt.title("Linear Regression Fit (from scratch)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    y_hat_ne_sorted = y_hat_ne[order]
    plt.figure()
    plt.scatter(X_sorted, y_sorted, s=10)
    plt.plot(X_sorted, y_hat_sorted, label="Gradient Descent")
    plt.plot(X_sorted, y_hat_ne_sorted, linestyle="--", label="Normal Equation")
    plt.legend()
    plt.title("Linear Regression: GD vs Closed-form")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    

    # 5) Plot training loss history (sampled)
    plt.figure()
    plt.plot(np.arange(len(losses)) * 50, losses)
    plt.title("Training Loss (MSE) over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.show()


if __name__ == "__main__":
    main()
