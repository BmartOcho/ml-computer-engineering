from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

FEATURE_NAMES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]



def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)


def load_wine_quality(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
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

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot



def add_bias_column(X: np.ndarray) -> np.ndarray:
    ones = np.ones((X.shape[0], 1))
    return np.hstack([ones, X])


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def normalize_features(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    Xb = add_bias_column(X)
    theta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    return theta


def evaluate_over_splits(
    csv_path: str,
    name: str,
    splits: int = 10,
    test_size: float = 0.2,
    lr: float = 1e-2,
    epochs: int = 5000,
) -> None:
    X, y = load_wine_quality(csv_path)

    rmses: list[float] = []
    baselines: list[float] = []

    for split_seed in range(splits):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, seed=split_seed)

        X_train_n, X_mean, X_std = normalize_features(X_train)
        X_test_n = (X_test - X_mean) / X_std

        theta_gd, _ = train_linear_regression_gd(X_train_n, y_train, lr=lr, epochs=epochs)
        yhat_test = predict(X_test_n, theta_gd)

        rmse_test = float(np.sqrt(mean_squared_error(y_test, yhat_test)))

        baseline = float(np.mean(y_train))
        rmse_baseline = float(np.sqrt(np.mean((y_test - baseline) ** 2)))

        rmses.append(rmse_test)
        baselines.append(rmse_baseline)

    print(f"\n=== 10-split stability ({name}) ===")
    print("Test RMSE mean:", float(np.mean(rmses)))
    print("Test RMSE std :", float(np.std(rmses)))
    print("Baseline RMSE mean:", float(np.mean(baselines)))
    print("All test RMSEs:", [round(x, 4) for x in rmses])


def run_one_dataset(name: str, path: str, lr: float, epochs: int, plot: bool = False) -> None:
    # 1) Load
    X, y = load_wine_quality(path)

    # 2) Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)

    # 3) Normalize using TRAIN stats only
    X_train_n, X_mean, X_std = normalize_features(X_train)
    X_test_n = (X_test - X_mean) / X_std

    # 4) Train GD
    print(f"\nTraining GD for {name}: lr={lr}, epochs={epochs}")
    theta_gd, losses = train_linear_regression_gd(X_train_n, y_train, lr=lr, epochs=epochs)

    # 5) Eval GD
    yhat_train = predict(X_train_n, theta_gd)
    yhat_test = predict(X_test_n, theta_gd)

    rmse_train = float(np.sqrt(mean_squared_error(y_train, yhat_train)))
    rmse_test = float(np.sqrt(mean_squared_error(y_test, yhat_test)))

    baseline = float(np.mean(y_train))
    rmse_baseline = float(np.sqrt(np.mean((y_test - baseline) ** 2)))

    # 6) Closed-form sanity check
    theta_cf = least_squares_closed_form(X_train_n, y_train)
    yhat_test_cf = predict(X_test_n, theta_cf)
    rmse_test_cf = float(np.sqrt(mean_squared_error(y_test, yhat_test_cf)))

    max_pred_diff = float(np.max(np.abs(yhat_test - yhat_test_cf)))
    max_theta_diff = float(np.max(np.abs(theta_gd - theta_cf)))

    unique_vals, counts = np.unique(y_test.astype(int), return_counts=True)
    dist = {int(u): int(c) for u, c in zip(unique_vals, counts)}

    print("Top 5 |weights| (closed-form, normalized features):")
    w = theta_cf[1:, 0]
    top = np.argsort(np.abs(w))[::-1][:5]
    for i in top:
        print(f"  {i:2d} {FEATURE_NAMES[i]:<20} weight={w[i]: .4f}")

    print(f"\n=== Wine Quality ({name}) ===")
    print("RMSE baseline (predict mean):", rmse_baseline)
    print("RMSE train:", rmse_train)
    print("RMSE test :", rmse_test)
    print("RMSE test (closed-form):", rmse_test_cf)
    print("max |GD - CF| on test preds:", max_pred_diff)
    print("max |theta_gd - theta_cf|:", max_theta_diff)
    print("y_test distribution:", dist)
    print("R^2 test:", r2_score(y_test, yhat_test))


    # 7) Plots (optional)
    if plot:
        plt.figure()
        plt.scatter(y_test, yhat_test, s=18, alpha=0.7, label="Model")
        plt.axhline(baseline, linestyle=":", linewidth=1, label="Baseline (mean)")

        minv = min(y_test.min(), yhat_test.min())
        maxv = max(y_test.max(), yhat_test.max())
        plt.plot([minv, maxv], [minv, maxv], linestyle="--", linewidth=1, label="Perfect")

        plt.xlabel("y_test (true)")
        plt.ylabel("yhat_test (predicted)")
        plt.title(f"Wine Quality {name}: Test Predicted vs True")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(losses)
        plt.title(f"Training Loss (MSE) over Epochs — {name}")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.yscale("log")
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.show()


def main() -> None:
    LR = 1e-2
    EPOCHS = 20000  # single knob for GD training length

    set_seed(42)

    datasets = [
        ("Red", "data/winequality-red.csv"),
        ("White", "data/winequality-white.csv"),
    ]

    # Single-split report for both datasets
    for name, path in datasets:
        run_one_dataset(name, path, lr=LR, epochs=EPOCHS, plot=False)

    # Stability report (10 random splits) for both datasets
    for name, path in datasets:
        evaluate_over_splits(path, name=name, splits=10, lr=LR, epochs=EPOCHS)


if __name__ == "__main__":
    main()
