# Linear Regression from Scratch (Gradient Descent)

## Goal

Implement and train a linear regression model **without scikit-learn**, using gradient descent, and visualize training loss.

This is Week 1 of the Foundations track.

---

## What this includes

- Synthetic dataset generation: `y = 3x + 10 + noise`
- Linear regression model implemented with a bias term
- Gradient descent training on Mean Squared Error (MSE)
- Plots:
  - Data + learned regression line
  - MSE loss curve over epochs

---

## How it works (high-level)

We model:

`y_hat = b + w*x`

We minimize Mean Squared Error:

`MSE = (1/n) * Σ (y_hat - y)^2`

Gradient descent iteratively updates parameters:

`theta = theta - lr * gradient`

Where:

`gradient = (2/n) * X^T * (y_hat - y)`

---

## Run it

### Install dependencies

```bash
pip install -r requirements.txt
```
