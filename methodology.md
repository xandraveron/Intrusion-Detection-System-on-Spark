# Methodology

## Problem Statement

Network intrusion detection systems (IDS) must classify network traffic into attack categories in real time and at scale. This project investigates whether distributing deep learning training across a Spark cluster improves throughput while maintaining classification accuracy, compared to single-machine training and classical ML baselines.

---

## Dataset — NSL-KDD

The **NSL-KDD** dataset is an improved version of the KDD Cup 1999 dataset, correcting redundant records that skewed evaluation metrics. It contains 41 features describing individual network connections and a label identifying the traffic type.

**Label mapping (5-class):**

| Class | Attack types included |
|---|---|
| Normal | Normal traffic |
| DoS | back, land, neptune, pod, smurf, teardrop, mailbomb, apache2, processtable, udpstorm, worm |
| Probe | satan, ipsweep, nmap, portsweep, mscan, saint |
| R2L | guess\_passwd, ftp\_write, imap, phf, multihop, warezmaster, warezclient, spy, xlock, xsnoop, snmpguess, snmpgetattack, httptunnel, sendmail, named |
| U2R | buffer\_overflow, loadmodule, rootkit, perl, sqlattack, xterm, ps |

**Training file:** `KDDTrain+.txt`
**Test file:** `KDDTest+.txt`

---

## Preprocessing

1. The `difficulty` column is dropped (not a feature — it is a dataset annotation).
2. The `num_outbound_cmds` column is dropped (zero variance — all values are 0).
3. The `su_attempted` column has a known encoding error (value `2` should be `0`) — corrected.
4. Categorical features (`protocol_type`, `service`, `flag`) are encoded using **One-Hot Encoding**.
5. Numerical features are scaled to [0, 1] using **Min-Max normalization**.
6. String labels are integer-encoded with `LabelEncoder` for compatibility with Keras.

---

## Class Balancing — SMOTE

The NSL-KDD training set is heavily imbalanced (DoS and Normal dominate; U2R has very few samples). **SMOTE** (Synthetic Minority Oversampling Technique) is applied to the training set to generate synthetic samples for underrepresented classes, giving each class an equal number of training examples.

SMOTE is applied **after** the train/test split and **only** to the training set to prevent data leakage.

---

## Models

All deep learning models use:
- **Hidden units:** 80 per layer
- **Output:** softmax over 5 classes
- **Loss:** categorical cross-entropy
- **Optimizer:** Adam
- **Epochs:** 5 per partition (Spark) / 5 total (single-machine)
- **Batch size:** 256

### MLP (Multi-Layer Perceptron)
Fully connected feedforward network. Input is the flat feature vector. Number of hidden layers is configurable (1–3).

### RNN (Simple Recurrent Network)
The flat feature vector is reshaped to `(1, features)` to create a single-timestep sequence. Number of recurrent layers is configurable (1–3).

### LSTM (Long Short-Term Memory)
Same input reshaping as RNN, with LSTM cells instead of simple recurrent units.

---

## Distributed Training — FedAvg over Spark

`05_dl_spark_mlp.py` implements a **Federated Averaging (FedAvg)**-style training loop using Apache Spark:

1. The training set is shuffled and saved as a memory-mapped file accessible to all workers.
2. Spark partitions the range `[0, N_PARTITIONS)` and maps each partition ID to a worker.
3. Each worker loads its slice of data from the shared memory-mapped file and trains a local copy of the model for 5 epochs.
4. The driver collects all local weight arrays and computes a **weighted average** (weighted by partition size).
5. The aggregated global model is evaluated on the test set on the driver.

This approach simulates federated learning in a single-node Spark cluster, allowing exploration of how partition count and executor configuration affect training time and accuracy.

---

## Experiment Configurations

The grid covers three cluster layouts:

| Config | Workers | Cores/worker | Exec memory | Partitions | Executors |
|--------|---------|--------------|-------------|------------|-----------|
| A | 4 | 2 | 7 g | 4 | 2 |
| B | 2 | 4 | 15 g | 8 | 2 |
| C | 4 | 2 | 7 g | 8 | 4 |

Each configuration is run with all three model types (MLP, RNN, LSTM) and all three layer depths (1, 2, 3), giving **27 total runs**.
