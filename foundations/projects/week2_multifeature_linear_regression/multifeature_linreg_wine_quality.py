

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)

def load_wine_quality_red(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(csv_path, delimiter=";", skip_header=1)
    X = data[:, :-1]
    y = data[:, -1:].reshape(-1, 1)
    return X, y

def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    test_n = int(n * test_size)
    test_idx = idx[:test_n]
    train_idx = idx[test_n:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

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

def main() -> None:
    LR = 1e-2
    EPOCHS = 5000

    set_seed(42)

    # 1) Load real data
    X, y = load_wine_quality_red("data/winequality-red.csv")

    # 2) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)

    # 3) Normalize using TRAIN stats only (avoid leakage)
    X_train_n, X_mean, X_std = normalize_features(X_train)
    X_test_n = (X_test - X_mean) / X_std

    # 4) Train GD on train set
    theta_gd, losses = train_linear_regression_gd(X_train_n, y_train, lr=LR, epochs=EPOCHS)

    # 5) Evaluate on train + test
    yhat_train = predict(X_train_n, theta_gd)
    yhat_test = predict(X_test_n, theta_gd)

    mse_train = mean_squared_error(y_train, yhat_train)
    mse_test = mean_squared_error(y_test, yhat_test)

    rmse_train = float(np.sqrt(mse_train))
    rmse_test = float(np.sqrt(mse_test))

    # 6) Baseline: predict mean of y_train
    baseline = float(np.mean(y_train))
    rmse_baseline = float(np.sqrt(np.mean((y_test - baseline) ** 2)))

    print("\n=== Wine Quality (Red) ===")
    print("RMSE baseline (predict mean):", rmse_baseline)
    print("RMSE train:", rmse_train)
    print("RMSE test :", rmse_test)

    # Optional: compare GD vs closed-form on TRAIN (sanity check)
    theta_cf = least_squares_closed_form(X_train_n, y_train)
    yhat_test_cf = predict(X_test_n, theta_cf)
    rmse_test_cf = float(np.sqrt(mean_squared_error(y_test, yhat_test_cf)))
    print("RMSE test (closed-form):", rmse_test_cf)
    print("max |GD - CF| on test preds:", float(np.max(np.abs(yhat_test - yhat_test_cf))))
    unique, counts = np.unique(y_test.astype(int), return_counts=True)
    print("y_test distribution:", dict(zip(unique, counts)))

    # 7) Plot: predicted vs true on TEST
    plt.figure()
    plt.scatter(y_test, yhat_test, s=18, alpha=0.7, label="Model")
    plt.axhline(baseline, linestyle=":", linewidth=1, label="Baseline (mean)")
    minv = min(y_test.min(), yhat_test.min())
    maxv = max(y_test.max(), yhat_test.max())
    plt.plot([minv, maxv], [minv, maxv], linestyle="--", linewidth=1)
    plt.xlabel("y_test (true)")
    plt.ylabel("yhat_test (predicted)")
    plt.title("Wine Quality Red: Test Predicted vs True")
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
