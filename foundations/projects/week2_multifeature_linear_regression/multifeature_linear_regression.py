

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)


def make_sythetic_data(
    n: int = 250,
    d: int = 3,
    noise_std: float = 10.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Create sythetic multi-feature linear data:
    y = X @ w_true + b_true + noise

    Returns:
    X: shape (n, d)
    y: shape (n, 1)
    w_true: shape (d, 1)
    b_true: float
    """
    X = np.random.uniform(0, 50, size=(n, d)) # (n, d)
    w_true = np.array([[3.0], [2.0], [1.5]]) # (d, 1) <-- for d=3
    b_true = 10.0

    noise = np.random.normal(0, noise_std, size=(n, 1))
    y = X @ w_true + b_true + noise # (n, 1)

    return X, y, w_true, b_true
    


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
    Equivalent to the normal equation but avoids eXplicit matriX inverse.
    """
    Xb = add_bias_column(X)
    theta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    return theta


def denormalize_theta(theta_norm: np.ndarray, X_mean: np.ndarray, X_std: np.ndarray) -> np.ndarray:
    """
    Convert parameters learned on normalized X back to raw-X units.

    theta_norm: shape (d+1, 1) where:
      theta_norm[0] = b_norm
      theta_norm[1:] = w_norm (d, 1)

    If Xn = (X - mean) / std and y = b_norm + w_norm^T Xn,
    then in raw X units:
      w_raw = w_norm / std
      b_raw = b_norm - sum(w_norm * mean / std)
    """
    b_norm = theta_norm[0:1, :] #(1, 1)
    w_norm = theta_norm[1:, :] #(d, 1)

    mean = X_mean.reshape(-1, 1) # (d, 1)
    std = X_std.reshape(-1, 1) # (d, 1)

    w_raw = w_norm / std #(d, 1)
    b_raw = b_norm - (w_norm.T @ (mean / std))  # shape (1,1)

    b_raw_scalar = b_raw.item() # extract a real Python float
    return np.vstack([[b_raw_scalar], w_raw]) # (d+1, 1)
    


def main() -> None:
    set_seed(42)

    # 1) Data (RAW)
    X, y, w_true, b_true = make_sythetic_data(n=250, d=3, noise_std=10.0)

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

    print("\n=== Ground truth (raw units) ===")
    print("True w:", w_true.ravel())
    print("True b:", b_true)

    print("\n=== Recovered (raw units) ===")
    print("CF w:", theta_cf_raw[1:, 0])
    print("CF b:", theta_cf_raw[0, 0])
    print("GD w:", theta_gd_raw[1:, 0])
    print("GD b:", theta_gd_raw[0, 0])


    # 7) Plot: predicted vs true (works for any number of features)
    plt.figure()
    plt.scatter(y, y_hat_gd, s=28, label="GD", alpha=.85, c="#1f77b4", marker="o", linewidths=0)
    plt.scatter(y, y_hat_cf, s=45, label="Closed-form", alpha=.95, c="#d62728", marker="o", linewidths=0)
    print("max |GD - CF|:", np.max(np.abs(y_hat_gd - y_hat_cf)))
    plt.xlabel("y (true)")
    plt.ylabel("y_hat (predicted)")
    plt.title("Predicted vs True")
    minv = min(y.min(), y_hat_gd.min(), y_hat_cf.min())
    maxv = max(y.max(), y_hat_gd.max(), y_hat_cf.max())
    plt.plot([minv, maxv], [minv, maxv], linestyle="--", linewidth=1)
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
