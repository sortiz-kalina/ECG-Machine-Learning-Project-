"""
Sanity-check run for the from-scratch network on the Wisconsin Breast Cancer
dataset (ships with scikit-learn, so no download needed).

Steps:
  1. Gradient check  - proves the hand-written backprop is correct
  2. Train the network
  3. Evaluate with medical-style metrics (not just accuracy)
  4. Compare against a logistic regression baseline
"""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from nn import NeuralNet, gradient_check, classification_report

# --- data -------------------------------------------------------------------
data = load_breast_cancer()
# sklearn labels: 0 = malignant, 1 = benign. Flip so 1 = malignant (the case
# we most care about catching), which makes "sensitivity" mean what it should.
X, y = data.data, 1 - data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# standardize using TRAIN statistics only (using test stats would leak info)
mean, std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
X_train_s = (X_train - mean) / std
X_test_s = (X_test - mean) / std

print(f"train {X_train.shape}, test {X_test.shape}, "
      f"malignant rate {y.mean():.1%}\n")

# --- 1. gradient check --------------------------------------------------------
check_net = NeuralNet([X.shape[1], 16, 8, 1], seed=0)
err = gradient_check(check_net, X_train_s[:20], y_train[:20])
print(f"Gradient check max relative error: {err:.2e}  "
      f"({'PASS' if err < 1e-6 else 'FAIL'})\n")

# --- 2. train ---------------------------------------------------------------
net = NeuralNet([X.shape[1], 16, 8, 1], seed=0)
net.fit(X_train_s, y_train, epochs=200, batch_size=32, lr=1e-3,
        weight_decay=1e-3, X_val=X_test_s, y_val=y_test)

# --- 3. evaluate ------------------------------------------------------------
def show(name, report):
    print(f"\n{name}")
    for k, v in report.items():
        print(f"  {k:22s} {v if isinstance(v, dict) else round(v, 4)}")

show("From-scratch NumPy network", classification_report(
    y_test, net.predict_proba(X_test_s)))

# --- 4. baseline ------------------------------------------------------------
lr = LogisticRegression(max_iter=1000).fit(X_train_s, y_train)
show("Logistic regression baseline", classification_report(
    y_test, lr.predict_proba(X_test_s)[:, 1]))

net.save("weights.npz", mean=mean, std=std)
print("\nSaved weights.npz (includes standardization stats for the API).")
