"""
Linear Regression from Scratch (Gradient Descent + Closed-Form)

This script intentionally uses BOTH:
1) Gradient Descent (optimization-based)
2) Closed-form solution (normal equation / least squares)

Key lesson:
- If you normalize features for GD stability, you MUST be consistent:
  train, predict, and compare in the same feature space.
- If you want interpretable parameters (w, b) in the original units,
  you must convert them back.

Run:
  python linear_regression_gd.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)


def make_synthetic_data(n: int = 250, noise_std: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
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
    ones = np.ones((X.shape[0], 1))
    return np.hstack([ones, X])


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def normalize_features(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      Xn: normalized X
      X_mean: mean of X (shape (1, d))
      X_std: std of X (shape (1, d))
    """
    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True) + 1e-12
    Xn = (X - X_mean) / X_std
    return Xn, X_mean, X_std


def train_linear_regression_gd(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 1e-2,
    epochs: int = 5000,
) -> tuple[np.ndarray, list[float]]:
    """
    Gradient descent on MSE for linear regression.
    Works for X shape (n, d). Uses bias column internally.
    """
    Xb = add_bias_column(X)
    n, d1 = Xb.shape
    theta = np.zeros((d1, 1))
    losses: list[float] = []

    for _ in range(epochs):
        y_pred = Xb @ theta
        error = y_pred - y
        grad = (2.0 / n) * (Xb.T @ error)
        theta -= lr * grad

        losses.append(mean_squared_error(y, y_pred))

    return theta, losses


def predict(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return add_bias_column(X) @ theta


def least_squares_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Numerically stable closed-form via least squares.
    Equivalent to the normal equation but avoids explicit matrix inverse.
    """
    Xb = add_bias_column(X)
    theta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    return theta


def denormalize_theta(theta_norm: np.ndarray, X_mean: np.ndarray, X_std: np.ndarray) -> np.ndarray:
    """
    Convert parameters learned on normalized X back to raw-X units.

    If Xn = (X - mean) / std, and model is:
      y = b_norm + w_norm * Xn

    Then in raw X units:
      w_raw = w_norm / std
      b_raw = b_norm - (w_norm * mean / std)
    """
    b_norm = float(theta_norm[0, 0])
    w_norm = float(theta_norm[1, 0])

    mean = float(X_mean[0, 0])
    std = float(X_std[0, 0])

    w_raw = w_norm / std
    b_raw = b_norm - (w_norm * mean / std)

    return np.array([[b_raw], [w_raw]])


def main() -> None:
    set_seed(42)

    # 1) Data (RAW)
    X, y = make_synthetic_data(n=250, noise_std=10.0)

    # 2) Normalize for stable GD (but keep RAW X for plotting)
    Xn, X_mean, X_std = normalize_features(X)

    # 3) Train with GD on NORMALIZED features
    theta_gd = None
    theta_gd, losses = train_linear_regression_gd(Xn, y, lr=1e-2, epochs=5000)

    # 4) Closed-form on the SAME normalized features (fair comparison)
    theta_cf = least_squares_closed_form(Xn, y)

    # 5) Predictions (use normalized Xn because that's the feature space of both thetas)
    y_hat_gd = predict(Xn, theta_gd)
    y_hat_cf = predict(Xn, theta_cf)

    mse_gd = mean_squared_error(y, y_hat_gd)
    mse_cf = mean_squared_error(y, y_hat_cf)

    # 6) Convert parameters back to RAW X units for interpretability
    theta_gd_raw = denormalize_theta(theta_gd, X_mean, X_std)
    theta_cf_raw = denormalize_theta(theta_cf, X_mean, X_std)

    print("=== Closed-form (Least Squares) ===")
    print(f"RAW units: w={theta_cf_raw[1,0]:.4f}, b={theta_cf_raw[0,0]:.4f}")
    print(f"MSE: {mse_cf:.4f}")

    print("\n=== Gradient Descent ===")
    print(f"RAW units: w={theta_gd_raw[1,0]:.4f}, b={theta_gd_raw[0,0]:.4f}")
    print(f"MSE: {mse_gd:.4f}")

    print("\n=== Loss debug ===")
    print("len(losses):", len(losses))
    print("first 5 losses:", losses[:5])
    print("last 5 losses:", losses[-5:])
    print("any NaN:", np.isnan(losses).any())
    print("any inf:", np.isinf(losses).any())

    # 7) Plot: data + both fitted lines (plotted against RAW X for readability)
    order = np.argsort(X[:, 0])
    X_sorted = X[order]
    y_sorted = y[order]

    y_gd_sorted = y_hat_gd[order]
    y_cf_sorted = y_hat_cf[order]

    plt.figure()
    plt.scatter(X_sorted, y_sorted, s=10)
    plt.plot(X_sorted, y_gd_sorted, label="Gradient Descent")
    plt.plot(X_sorted, y_cf_sorted, linestyle="--", label="Closed-form")
    plt.title("Linear Regression Fit (GD vs Closed-form)")
    plt.xlabel("x (raw units)")
    plt.ylabel("y")
    plt.legend()
    plt.show()

    # 8) Plot: loss curve
    plt.figure()
    plt.plot(losses)
    plt.title("Training Loss (MSE) over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.yscale("log")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.show()


if __name__ == "__main__":
    main()
