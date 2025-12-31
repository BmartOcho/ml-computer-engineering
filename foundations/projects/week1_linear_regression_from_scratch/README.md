# Linear Regression from Scratch (Gradient Descent)

## Goal

Implement and train a linear regression model **without scikit-learn**, using gradient descent, and visualize the learning process.

This is **Week 1 of the Foundations track**.

---

## What this includes

- Synthetic dataset generation: `y = 3x + 10 + noise`
- Linear regression model implemented with an explicit bias term
- Gradient descent optimization using Mean Squared Error (MSE)
- Closed-form (least squares) solution for comparison
- Plots:
  - Data with fitted regression lines
  - Training loss (MSE) over epochs

---

## How it works (high-level)

The model assumes a linear relationship:

`ŷ = b + w·x`

It learns parameters by minimizing Mean Squared Error:

`MSE = (1/n) · Σ (ŷ − y)²`

Gradient descent updates parameters iteratively:

`θ ← θ − α · ∇MSE`

Where:

`∇MSE = (2/n) · Xᵀ(ŷ − y)`

The model does not “understand” the data — it only receives feedback on how wrong its predictions are and adjusts parameters to reduce that error.

---

## Running the code

### Install dependencies

```bash
pip install -r requirements.txt
```
