# Intrusion Detection with Apache Spark & Deep Learning

## Federated deep learning for network intrusion detection — MLP, RNN, and LSTM models distributed with Apache Spark on NSL-KDD dataset.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Running the Code](#running-the-code)
  - [Step-by-step](#step-by-step)
  - [Full Spark Grid (automated)](#full-spark-grid-automated)
- [Configuration Reference](#configuration-reference)
- [Project Structure](#project-structure)
- [References](#references)
  

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.9 or higher |
| Java (JDK) | 11 (recommended: Eclipse Adoptium) |
| Apache Spark | 3.5.x with Hadoop 3 |
| OS | Windows 10/11 (scripts use PowerShell) |

> Java 11 is required. Spark 3.5.x is not compatible with Java 17+ in all configurations.

---

## Setup

**1. Clone the repository**

```bash
git clone <repo-url>
cd intrusion-detection-apache-spark
```

**2. Create and activate a virtual environment**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**3. Install dependencies**

```powershell
pip install numpy pandas scikit-learn imbalanced-learn tensorflow pyspark joblib
```

**4. Set environment variables**

Set the following before running any Spark script, adjusted to your installation paths:

```powershell
$env:SPARK_HOME  = "C:\spark\spark-3.5.x-bin-hadoop3"
$env:JAVA_HOME   = "C:\Program Files\Eclipse Adoptium\jdk-11.x.x-hotspot"
$env:PATH        = "$env:JAVA_HOME\bin;$env:SPARK_HOME\bin;$env:PATH"
```

Alternatively, pass these paths as parameters to `setEnvAndRun.ps1` (see [Configuration Reference](#configuration-reference)).

---

## Dataset

This project uses the **NSL-KDD** dataset. It is not included in the repository and must be downloaded manually.

**Download:** https://www.unb.ca/cic/datasets/nsl.html

**Required files** — place them in the `nsl-kdd/` folder at the repo root:

```
nsl-kdd/
├── KDDTrain+.txt
└── KDDTest+.txt
```

Only these two files are needed. The `.arff` variants and partial splits are not used.

---

## Pipeline

The project runs as a sequential 5-step pipeline. Each script depends on the outputs of the previous one.

| Step | Script | Description |
|------|--------|-------------|
| 1 | `01_preprocess.py` | Loads NSL-KDD, maps 40+ attack types to 5 classes, applies One-Hot encoding and Min-Max scaling. Saves artifacts to `artifacts/`. |
| 2 | `02_smote.py` | Applies SMOTE oversampling to the training set to correct class imbalance. |
| 3 | `03_baseline.py` | Trains and evaluates classical ML baselines: Decision Tree, KNN, and Linear SVM. |
| 4 | `04_dl_nospark.py` | Trains MLP, RNN, and LSTM models using TensorFlow without Spark (single-machine baseline). |
| 5 | `05_dl_spark_mlp.py` | Distributes training across a Spark cluster using a FedAvg-style approach. Appends results to `results/spark_results.csv`. |

---

## Running the Code

### Step-by-step

Run each script in order from the repo root:

```powershell
python 01_preprocess.py
python 02_smote.py
python 03_baseline.py
python 04_dl_nospark.py
```

For the Spark script, run it through `spark-submit` (handled automatically by the PowerShell script):

```powershell
.\scripts\setEnvAndRun.ps1 -Mode single -PyPath ".\.venv\Scripts\python.exe"
```

### Full Spark Grid (automated)

To run the complete grid of configurations (3 cluster layouts × 3 models × 3 layer depths = 27 runs):

```powershell
.\scripts\setEnvAndRun.ps1 -Mode full-grid -PyPath ".\.venv\Scripts\python.exe" -DriverMem "4g"
```

The script automatically starts and stops the Spark cluster between configuration groups. Results are appended to `results/spark_results.csv`.

---

## Configuration Reference

`scripts/setEnvAndRun.ps1` accepts the following parameters:

| Parameter | Default | Description |
|---|---|---|
| `-Mode` | `single` | Run mode: `single`, `grid`, or `full-grid` |
| `-PyPath` | `$env:PYSPARK_PYTHON` | Path to your venv `python.exe` |
| `-SparkHome` | `$env:SPARK_HOME` | Path to your Spark installation |
| `-JavaHome` | `$env:JAVA_HOME` | Path to your JDK 11 installation |
| `-DriverMem` | `6g` | Spark driver memory |
| `-ExecMem` | `4g` | Spark executor memory (single/grid modes) |
| `-ExecCores` | `1` | Cores per executor (single/grid modes) |
| `-NumExec` | `2` | Number of executors (single/grid modes) |
| `-Partitions` | `6` | Number of data partitions (single mode) |
| `-Layers` | `2` | Number of hidden layers (single mode) |
| `-ModelType` | `mlp` | Model architecture: `mlp`, `rnn`, or `lstm` |

**Example — single run with a specific model:**

```powershell
.\scripts\setEnvAndRun.ps1 `
  -Mode single `
  -ModelType lstm `
  -Layers 3 `
  -Partitions 8 `
  -PyPath ".\.venv\Scripts\python.exe" `
  -SparkHome "C:\spark\spark-3.5.7-bin-hadoop3" `
  -JavaHome "C:\Program Files\Eclipse Adoptium\jdk-11.0.28.6-hotspot"
```

---

## Project Structure

```
├── 01_preprocess.py          # Step 1 — feature engineering & encoding
├── 02_smote.py               # Step 2 — class balancing
├── 03_baseline.py            # Step 3 — classical ML baselines
├── 04_dl_nospark.py          # Step 4 — deep learning without Spark
├── 05_dl_spark_mlp.py        # Step 5 — distributed training with Spark
├── scripts/
│   └── setEnvAndRun.ps1      # Spark cluster manager & experiment runner
├── nsl-kdd/                  # Dataset files (not tracked — download manually)
├── artifacts/                # Generated by pipeline (not tracked)
├── results/
│   └── spark_results.csv     # Experiment results log
└── docs/
    ├── methodology.md        # Design decisions and approach
    └── results.md            # Results analysis and tables
```

## References

This project is based on and inspired by:

> M. Haggag, M. M. Tantawy and M. M. S. El-Soudani, "Implementing a Deep Learning Model for Intrusion Detection on Apache Spark Platform," *IEEE Access*, vol. 8, pp. 163660-163672, 2020. doi: [10.1109/ACCESS.2020.3019931](https://doi.org/10.1109/ACCESS.2020.3019931)

**Dataset:**
> M. Tavallaee, E. Bagheri, W. Lu, and A. Ghorbani, "A Detailed Analysis of the KDD CUP 99 Data Set," *Proc. IEEE Symp. Computational Intelligence in Security and Defense Applications (CISDA)*, 2009.
> Download: https://www.unb.ca/cic/datasets/nsl.html
