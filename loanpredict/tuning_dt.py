#for tuning decision tree
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from model.preprocess import load_data
from model.decision_tree import train_decision_tree, evaluate

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

depth_values = [ 2, 3, 4, 5, 6, 7, 8, 9, 10]
results = []

for depth in depth_values:
    model = train_decision_tree(X_train, y_train, depth_value = depth, minleaf_value = 7)
    result = evaluate(model, X_train, y_train, X_test, y_test)

    results.append({
        "max_depth": depth,
        "Train Accuracy": result["train_acc"],
        "Test Accuracy": result["test_acc"],
        "Overfitting Gap": result["overfit_gap"],
        "Test F1-macro": result["test_f1"]
    })
    
df_result = pd.DataFrame(results)

plt.figure(figsize=(7, 5))

plt.plot(
    df_result["max_depth"].to_numpy(),
    df_result["Train Accuracy"].to_numpy(),
    marker="o",
    label="Train Accuracy"
)

plt.plot(
    df_result["max_depth"].to_numpy(),
    df_result["Test Accuracy"].to_numpy(),
    marker="s",
    label="Test Accuracy"
)

plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree: Effect of Max Depth on Overfitting")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
