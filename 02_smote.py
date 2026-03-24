import numpy as np
from pathlib import Path
from collections import Counter
from imblearn.over_sampling import SMOTE

BASE_DIR = Path(__file__).resolve().parent
ART = BASE_DIR / "artifacts"

def main():
    if not (ART / "X_train.npy").exists():
        raise FileNotFoundError("Missing artifacts from preprocessing. Run 01_preprocess.py first.")

    X_train = np.load(ART / "X_train.npy")
    y_train = np.load(ART / "y_train.npy")   # already integer-encoded by 01_preprocess.py

    print("\nBefore SMOTE:")
    print(Counter(y_train))

    smote = SMOTE(random_state=42)
    X_sm, y_sm = smote.fit_resample(X_train, y_train)

    print("\nAfter SMOTE:")
    print(Counter(y_sm))

    # Save
    np.save(ART / "X_train_sm.npy", X_sm)
    np.save(ART / "y_train_sm.npy", y_sm)

    print("\nSaved: artifacts/X_train_sm.npy, artifacts/y_train_sm.npy")
    print("SMOTE train shape:", X_sm.shape)

if __name__ == "__main__":
    main()
