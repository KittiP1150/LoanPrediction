
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from model.preprocess import load_data
from model.svc import train_svc, evaluate as eval_svc
from model.logistic_regression import train_logistic, evaluate as eval_log
from model.decision_tree import train_decision_tree, evaluate as eval_tree


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "loan_data_set.csv")

# load data
df = load_data(DATA_PATH)
X = df.drop(columns=["Loan_Status"])
y = df["Loan_Status"]

# split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Logistic Regression
start = time.perf_counter()
log_model = train_logistic(X_train, y_train, 0.1, penalty_type= "l1", solver_type= "liblinear" )
train_time_log = time.perf_counter() - start

start = time.perf_counter()
log_model.predict(X_test)
infer_time_log = time.perf_counter() - start

log_result = eval_log(log_model, X_train, y_train, X_test, y_test)

# SVC
start = time.perf_counter()
svc_model = train_svc(X_train, y_train, C_value= 1, gamma_value= 0.01)
train_time_svc = time.perf_counter() - start

start = time.perf_counter()
svc_model.predict(X_test)
infer_time_svc = time.perf_counter() - start

svc_result = eval_svc(svc_model, X_train, y_train, X_test, y_test)

# Decision Tree
start = time.perf_counter()
tree_model = train_decision_tree(X_train, y_train, depth_value = 5, minleaf_value = 7)
train_time_tree = time.perf_counter() - start

start = time.perf_counter()
tree_model.predict(X_test)
infer_time_tree = time.perf_counter() - start

tree_result = eval_tree(tree_model, X_train, y_train, X_test, y_test)


# compare results
compare_df = pd.DataFrame([
    {
        "Model": "Logistic Regression",
        "Train Accuracy": log_result["train_acc"],
        "Test Accuracy": log_result["test_acc"],
        "Training Time (s)": train_time_log,
        "Inference Time (s)": infer_time_log
    },
    {
        "Model": "Decision Tree",
        "Train Accuracy": tree_result["train_acc"],
        "Test Accuracy": tree_result["test_acc"],
        "Training Time (s)": train_time_tree,
        "Inference Time (s)": infer_time_tree
    },
    {
        "Model": "SVC (RBF)",
        "Train Accuracy": svc_result["train_acc"],
        "Test Accuracy": svc_result["test_acc"],
        "Training Time (s)": train_time_svc,
        "Inference Time (s)": infer_time_svc
    }
])

plt.figure(figsize=(7,4))
plt.bar(compare_df["Model"], compare_df["Training Time (s)"])
plt.ylabel("Seconds")
plt.title("Training Time Comparison")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,4))
plt.bar(compare_df["Model"], compare_df["Inference Time (s)"])
plt.ylabel("Seconds")
plt.title("Inference Time Comparison")
plt.tight_layout()
plt.show()

models = compare_df["Model"]
train_acc = compare_df["Train Accuracy"]
test_acc = compare_df["Test Accuracy"]

x = np.arange(len(models))
width = 0.35

plt.figure(figsize=(8, 5))

plt.bar(x - width/2, train_acc, width, label="Train Accuracy")
plt.bar(x + width/2, test_acc, width, label="Test Accuracy")

plt.xticks(x, models, rotation=10)
plt.ylabel("Accuracy")
plt.title("Train vs Test Accuracy Comparison")
plt.ylim(0.6, 0.9)
plt.legend()

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Logistic Regression
cm_log = confusion_matrix(y_test, log_model.predict(X_test))
disp_log = ConfusionMatrixDisplay(
    confusion_matrix=cm_log,
    display_labels=["Rejected (N)", "Approved (Y)"]
)
disp_log.plot(ax=axes[0], cmap="Blues", values_format="d")
axes[0].set_title("Logistic Regression")

# Decision Tree
cm_tree = confusion_matrix(y_test, tree_model.predict(X_test))
disp_tree = ConfusionMatrixDisplay(
    confusion_matrix=cm_tree,
    display_labels=["Rejected (N)", "Approved (Y)"]
)
disp_tree.plot(ax=axes[1], cmap="Blues", values_format="d")
axes[1].set_title("Decision Tree")

# SVC (RBF)
cm_svm = confusion_matrix(y_test, svc_model.predict(X_test))
disp_svm = ConfusionMatrixDisplay(
    confusion_matrix=cm_svm,
    display_labels=["Rejected (N)", "Approved (Y)"]
)
disp_svm.plot(ax=axes[2], cmap="Blues", values_format="d")
axes[2].set_title("SVC (RBF)")

plt.suptitle("Confusion Matrix Comparison Across Models", fontsize=14)
plt.tight_layout()
plt.show()