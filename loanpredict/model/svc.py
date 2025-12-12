from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from .preprocess import build_preprocess_pipeline

def train_svc(X_train, y_train, C_value, gamma_value):
    preprocess = build_preprocess_pipeline()

    model = Pipeline([
        ("preprocess", preprocess),
        ("svm", SVC(
            kernel="rbf",
            C=C_value,
            gamma=gamma_value,
            class_weight="balanced",
            probability=True,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)
    return model


def evaluate(model, X_train, y_train, X_test, y_test):
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    train_f1 = f1_score(y_train, model.predict(X_train), average="macro")
    test_f1 = f1_score(y_test, model.predict(X_test), average="macro")

    return {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "overfit_gap": train_acc - test_acc,
        "train_f1": train_f1,
        "test_f1": test_f1
    }