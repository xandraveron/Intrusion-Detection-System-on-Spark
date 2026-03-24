# src/03_baseline.py
import numpy as np
import time
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent
ART = BASE_DIR / "artifacts"

def run_model(name, model, X_train, y_train, X_test, y_test):
    print(f"\n=== {name} ===")
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Training time: {train_time:.2f} sec")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    return train_time, acc

def main():
    if not (ART / "X_train_sm.npy").exists():
        raise FileNotFoundError("Missing SMOTE artifacts. Run 02_smote.py first.")

    X_train = np.load(ART / "X_train_sm.npy")
    y_train = np.load(ART / "y_train_sm.npy", allow_pickle=True)

    X_test = np.load(ART / "X_test.npy")
    y_test = np.load(ART / "y_test.npy", allow_pickle=True)

    results = {}

    results["DecisionTree"] = run_model(
        "Decision Tree",
        DecisionTreeClassifier(random_state=42),
        X_train, y_train, X_test, y_test
    )

    results["KNN"] = run_model(
        "K-Nearest Neighbors",
        KNeighborsClassifier(n_neighbors=5),
        X_train, y_train, X_test, y_test
    )


    results["SVM"] = run_model(
        "Linear SVM",
        LinearSVC(random_state=42, max_iter=5000),
        X_train, y_train, X_test, y_test
    )

    print("\nSummary (training time, accuracy):")
    for k, v in results.items():
        print(k, v)

if __name__ == "__main__":
    main()
