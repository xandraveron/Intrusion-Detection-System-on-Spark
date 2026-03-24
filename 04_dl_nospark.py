# src/04_dl_nospark.py
import numpy as np
import time
from pathlib import Path
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN
from tensorflow.keras.utils import to_categorical

BASE_DIR = Path(__file__).resolve().parent
ART = BASE_DIR / "artifacts"

EPOCHS = 5
BATCH_SIZE = 256

def load_data():
    if not (ART / "X_train_sm.npy").exists():
        raise FileNotFoundError("Missing SMOTE artifacts. Run 02_smote.py first.")

    X_train = np.load(ART / "X_train_sm.npy")
    y_train = np.load(ART / "y_train_sm.npy", allow_pickle=True)
    X_test  = np.load(ART / "X_test.npy")
    y_test  = np.load(ART / "y_test.npy", allow_pickle=True)

    # Labels are already integer-encoded by 01_preprocess.py
    y_train_enc = y_train
    y_test_enc  = y_test

    y_train_cat = to_categorical(y_train_enc)

    return X_train, y_train_cat, X_test, y_test_enc

def build_mlp(input_dim, num_classes):
    model = Sequential([
        Dense(80, activation="relu", input_shape=(input_dim,)),
        Dense(80, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def build_rnn(input_dim, num_classes):
    model = Sequential([
        SimpleRNN(80, activation="tanh", input_shape=(1, input_dim)),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def build_lstm(input_dim, num_classes):
    model = Sequential([
        LSTM(80, activation="tanh", input_shape=(1, input_dim)),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def run_model(name, model, X_train, y_train, X_test, y_test):
    print(f"\n=== {name} (no Spark) ===")
    start = time.time()

    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    train_time = time.time() - start

    y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
    acc = accuracy_score(y_test, y_pred)

    print(f"Training time: {train_time:.2f} sec")
    print(f"Accuracy: {acc:.4f}")

    return train_time, acc

def main():
    X_train, y_train, X_test, y_test = load_data()
    input_dim = X_train.shape[1]
    num_classes = y_train.shape[1]

    # MLP
    run_model(
        "MLP",
        build_mlp(input_dim, num_classes),
        X_train, y_train, X_test, y_test
    )

    # RNN / LSTM need 3D input: (samples, timesteps=1, features)
    X_train_seq = X_train.reshape((X_train.shape[0], 1, input_dim))
    X_test_seq  = X_test.reshape((X_test.shape[0], 1, input_dim))

    run_model(
        "RNN",
        build_rnn(input_dim, num_classes),
        X_train_seq, y_train, X_test_seq, y_test
    )

    run_model(
        "LSTM",
        build_lstm(input_dim, num_classes),
        X_train_seq, y_train, X_test_seq, y_test
    )

if __name__ == "__main__":
    main()
