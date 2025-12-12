#for tuning svc
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from model.preprocess import load_data
from model.svc import train_svc, evaluate

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

C_values = [0.001, 0.01, 0.1, 1, 10, 100]
gamma_values = [0.001, 0.01, 0.1, "scale"]

results = []

for C in C_values:
    for gamma in gamma_values:
        model = train_svc(X_train, y_train, C, gamma)
        result = evaluate(model, X_train, y_train, X_test, y_test)

        results.append({
            "C": C,
            "gamma": gamma,
            "Train Accuracy": result["train_acc"],
            "Test Accuracy": result["test_acc"],
            "Overfit Gap": result["overfit_gap"],
            "Test F1-macro": result["test_f1"]
        })

df_result = pd.DataFrame(results)

plt.figure(figsize=(7,5))

for gamma in gamma_values:
    subset = df_result[df_result["gamma"] == gamma]

    plt.plot(
        subset["C"].to_numpy(),
        subset["Test Accuracy"].to_numpy(),
        marker="o",
        label=f"gamma={gamma}"
    )

plt.xscale("log")
plt.xlabel("C (log scale)")
plt.ylabel("Test Accuracy")
plt.title("SVC (RBF): Effect of C and Gamma")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
