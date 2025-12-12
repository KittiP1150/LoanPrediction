#for tuning logistic regression
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from model.preprocess import load_data
from model.logistic_regression import train_logistic, evaluate
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


DATA_PATH = os.path.join(BASE_DIR, "data", "loan_data_set.csv")
df = load_data(DATA_PATH)
X = df.drop(columns=["Loan_Status"])
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

C_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
penalties = ["l1", "l2"]

results = []

for C in C_values:
    for penalty in penalties:
        solver = "liblinear" if penalty == "l1" else "lbfgs"

        model = train_logistic(X_train, y_train, C, penalty, solver)
        result = evaluate(model, X_train, y_train, X_test, y_test)

        results.append({
            "C": C,
            "Penalty": penalty,
            "Train Accuracy": result["train_acc"],
            "Test Accuracy": result["test_acc"],
            "Overfit Gap": result["overfit_gap"],
            "Test F1-macro": result["test_f1"]
        })

df_result = pd.DataFrame(results)
df_l1 = df_result[df_result["Penalty"] == "l1"]
df_l2 = df_result[df_result["Penalty"] == "l2"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

axes[0].plot(
    df_l1["C"].to_numpy(),
    df_l1["Train Accuracy"].to_numpy(),
    marker="o",
    label="Train Accuracy"
)
axes[0].plot(
    df_l1["C"].to_numpy(),
    df_l1["Test Accuracy"].to_numpy(),
    marker="s",
    label="Test Accuracy"
)

axes[0].set_xscale("log")
axes[0].set_xlabel("C (log scale)")
axes[0].set_ylabel("Accuracy")
axes[0].set_title("Logistic Regression (L1)")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(
    df_l2["C"].to_numpy(),
    df_l2["Train Accuracy"].to_numpy(),
    marker="o",
    label="Train Accuracy"
)
axes[1].plot(
    df_l2["C"].to_numpy(),
    df_l2["Test Accuracy"].to_numpy(),
    marker="s",
    label="Test Accuracy"
)

axes[1].set_xscale("log")
axes[1].set_xlabel("C (log scale)")
axes[1].set_title("Logistic Regression (L2)")
axes[1].legend()
axes[1].grid(True)

plt.suptitle("Logistic Regression: Train vs Test Accuracy (L1 vs L2)", fontsize=14)
plt.tight_layout()
plt.show()

