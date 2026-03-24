# src/05_dl_spark_mlp.py
import os
import sys
import time
import csv
import datetime
from pathlib import Path

import numpy as np
from pyspark.sql import SparkSession
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from tensorflow.keras.utils import to_categorical

BASE_DIR = Path(__file__).resolve().parent
ART = BASE_DIR / "artifacts"
MM_DIR = ART / "mm"
MM_DIR.mkdir(parents=True, exist_ok=True)

X_PATH = MM_DIR / "X_train_sm.npy"
Y_PATH = MM_DIR / "y_train_cat.npy"
RESULTS_PATH = Path(r"D:\MASTERS DS\An1Sem1\Big Data\Intrusion detection apache spark\results\spark_results.csv")

EPOCHS     = 5
BATCH_SIZE = 256
PARTITIONS  = int(sys.argv[1])    if len(sys.argv) > 1 else 2
N_HIDDEN    = int(sys.argv[2])    if len(sys.argv) > 2 else 2
MODEL_TYPE  = sys.argv[3].lower() if len(sys.argv) > 3 else "mlp"   # mlp | rnn | lstm

def append_result_row(row: dict, csv_path: Path = RESULTS_PATH):
    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    write_header = not p.exists()
    with p.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)

def ensure_memmap(path: Path, arr: np.ndarray):
    if path.exists():
        try:
            existing = np.load(str(path), mmap_mode="r")
            if existing.shape == arr.shape:
                return
        except Exception:
            pass
    np.save(path, arr)

# ── Model builders ────────────────────────────────────────────────────────────

def build_mlp(input_dim, num_classes, n_hidden=2, hidden_units=80):
    layers = [Dense(hidden_units, activation="relu", input_shape=(input_dim,))]
    for _ in range(n_hidden - 1):
        layers.append(Dense(hidden_units, activation="relu"))
    layers.append(Dense(num_classes, activation="softmax"))
    model = Sequential(layers)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def build_rnn(input_dim, num_classes, n_hidden=2, hidden_units=80):
    # Input shape: (timesteps=1, features)
    layers = [SimpleRNN(hidden_units, activation="tanh", input_shape=(1, input_dim),
                        return_sequences=(n_hidden > 1))]
    for i in range(n_hidden - 2):
        layers.append(SimpleRNN(hidden_units, activation="tanh",
                                return_sequences=(i < n_hidden - 3)))
    if n_hidden > 1:
        layers.append(SimpleRNN(hidden_units, activation="tanh"))
    layers.append(Dense(num_classes, activation="softmax"))
    model = Sequential(layers)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def build_lstm(input_dim, num_classes, n_hidden=2, hidden_units=80):
    # Input shape: (timesteps=1, features)
    layers = [LSTM(hidden_units, activation="tanh", input_shape=(1, input_dim),
                   return_sequences=(n_hidden > 1))]
    for i in range(n_hidden - 2):
        layers.append(LSTM(hidden_units, activation="tanh",
                           return_sequences=(i < n_hidden - 3)))
    if n_hidden > 1:
        layers.append(LSTM(hidden_units, activation="tanh"))
    layers.append(Dense(num_classes, activation="softmax"))
    model = Sequential(layers)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def build_model(model_type, input_dim, num_classes, n_hidden):
    if model_type == "rnn":
        return build_rnn(input_dim, num_classes, n_hidden)
    if model_type == "lstm":
        return build_lstm(input_dim, num_classes, n_hidden)
    return build_mlp(input_dim, num_classes, n_hidden)

# ── Partition training (runs on each Spark worker) ────────────────────────────

def train_one_partition(part_id, input_dim, num_classes, global_weights,
                        n_hidden, model_type, local_epochs=5):
    import numpy as np
    import tensorflow as tf

    X = np.load(str(X_PATH), mmap_mode="r")
    Y = np.load(str(Y_PATH), mmap_mode="r")

    n     = X.shape[0]
    start = int(part_id * n / PARTITIONS)
    end   = int((part_id + 1) * n / PARTITIONS)

    X_part = np.array(X[start:end])
    y_part = np.array(Y[start:end])

    part_n = len(X_part)
    if part_n == 0:
        return None

    # RNN / LSTM need 3-D input: (samples, timesteps=1, features)
    if model_type in ("rnn", "lstm"):
        X_part = X_part.reshape((part_n, 1, input_dim))

    model = build_model(model_type, input_dim, num_classes, n_hidden)
    model.set_weights(global_weights)
    model.fit(X_part, y_part, epochs=local_epochs, batch_size=BATCH_SIZE, verbose=0)

    return (model.get_weights(), part_n)

# ── Driver ────────────────────────────────────────────────────────────────────

def main():
    builder = SparkSession.builder.appName(f"DLS-IDS-{MODEL_TYPE.upper()}-Spark")
    spark_local_dir = os.getenv("SPARK_LOCAL_DIR")
    if spark_local_dir:
        builder = builder.config("spark.local.dir", spark_local_dir)
    spark = builder.getOrCreate()
    sc = spark.sparkContext

    if not (ART / "X_train_sm.npy").exists():
        raise FileNotFoundError("Missing SMOTE artifacts. Run 02_smote.py first.")

    X_train = np.load(ART / "X_train_sm.npy")
    y_train = np.load(ART / "y_train_sm.npy", allow_pickle=True)
    X_test  = np.load(ART / "X_test.npy")
    y_test  = np.load(ART / "y_test.npy", allow_pickle=True)

    y_train_enc = y_train   # already integers from 01_preprocess.py
    y_test_enc  = y_test

    y_train_cat = to_categorical(y_train_enc)
    num_classes = y_train_cat.shape[1]
    input_dim   = X_train.shape[1]

    # Shuffle so each partition gets a balanced class distribution
    rng = np.random.default_rng(42)
    idx = rng.permutation(X_train.shape[0])
    X_train     = X_train[idx]
    y_train_cat = y_train_cat[idx]

    ensure_memmap(X_PATH, X_train)
    ensure_memmap(Y_PATH, y_train_cat)

    rdd = sc.parallelize(range(PARTITIONS), PARTITIONS).cache()
    rdd.count()

    print(f"\n=== {MODEL_TYPE.upper()} (Spark, FedAvg, partitions={PARTITIONS}, layers={N_HIDDEN}) ===")
    t0 = time.time()

    global_model   = build_model(MODEL_TYPE, input_dim, num_classes, N_HIDDEN)
    global_weights = global_model.get_weights()

    results = (
        rdd.map(lambda pid: train_one_partition(
                    pid, input_dim, num_classes, global_weights,
                    N_HIDDEN, MODEL_TYPE, local_epochs=EPOCHS))
           .filter(lambda x: x is not None)
           .collect()
    )

    print("Partitions returned:", len(results))
    print("Samples per partition:", [n_i for (_, n_i) in results])

    if not results:
        raise RuntimeError("No partitions returned samples.")
    total_n = sum(n_i for (_, n_i) in results)

    # Weighted-average weights (FedAvg)
    avg_weights = None
    for (w_i, n_i) in results:
        if avg_weights is None:
            avg_weights = [layer * (n_i / total_n) for layer in w_i]
        else:
            for j in range(len(avg_weights)):
                avg_weights[j] += w_i[j] * (n_i / total_n)

    train_time = time.time() - t0

    # Evaluate on driver
    global_model.set_weights(avg_weights)
    X_test_eval = X_test.reshape((X_test.shape[0], 1, input_dim)) \
                  if MODEL_TYPE in ("rnn", "lstm") else X_test
    y_pred = global_model.predict(X_test_eval, verbose=0).argmax(axis=1)
    acc    = accuracy_score(y_test_enc, y_pred)

    ts   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conf = spark.sparkContext.getConf()
    row  = {
        "timestamp":          ts,
        "model":              MODEL_TYPE,
        "partitions":         PARTITIONS,
        "layers":             N_HIDDEN,
        "master":             conf.get("spark.master", ""),
        "executor_memory":    conf.get("spark.executor.memory", ""),
        "executor_cores":     conf.get("spark.executor.cores", ""),
        "executor_instances": conf.get("spark.executor.instances", ""),
        "driver_memory":      conf.get("spark.driver.memory", ""),
        "train_time_s":       round(train_time, 4),
        "accuracy":           round(acc, 6),
    }
    append_result_row(row)
    print(f"Saved results to {RESULTS_PATH}")
    print(f"File exists: {RESULTS_PATH.exists()}  |  Size: {RESULTS_PATH.stat().st_size} bytes")
    print(f"\nTraining time (Spark): {train_time:.2f} sec")
    print(f"Accuracy: {acc:.4f}")

    spark.stop()

if __name__ == "__main__":
    main()
