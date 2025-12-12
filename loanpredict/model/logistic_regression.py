from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from .preprocess import build_preprocess_pipeline


def train_logistic(X_train, y_train, C_value, penalty_type, solver_type):
    preprocess = build_preprocess_pipeline()


    model = Pipeline([
        ("preprocess", preprocess),
        ("logreg", LogisticRegression(
            C=C_value, 
            penalty=penalty_type,
            solver=solver_type,
            max_iter=1000,
            class_weight="balanced"
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