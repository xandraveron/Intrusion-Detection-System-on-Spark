# Results

---

## Baseline Models (no Spark)

Results from `03_baseline.py` and `04_dl_nospark.py`.

| Model | Train Time (s) | Accuracy |
|---|---|---|
| Decision Tree | 11.40 | 0.7394 |
| KNN (k=5) | 0.07 | 0.7660 |
| Linear SVM | 19.79 | 0.7852 |
| MLP (2 layers, no Spark) | 12.61 | 0.8039 |
| RNN (2 layers, no Spark) | 12.86 | 0.7832 |
| LSTM (2 layers, no Spark) | 22.87 | 0.7726 |

---

## Spark Grid Results

Results from `05_dl_spark_mlp.py` via `setEnvAndRun.ps1 -Mode full-grid`.

Three cluster configurations were tested:

| Config | Partitions | Executors | Cores/Exec | Exec Memory |
|---|---|---|---|---|
| A | 4 | 2 | 2 | 7g |
| B | 8 | 2 | 4 | 15g |
| C | 8 | 4 | 2 | 7g |

### Config A — 4 partitions, 2 executors × 2 cores (7g)

| Model | Layers | Train Time (s) | Accuracy |
|---|---|---|---|
| MLP | 1 | 15.69 | 0.7995 |
| MLP | 2 | 16.08 | 0.7943 |
| MLP | 3 | 18.67 | 0.7797 |
| RNN | 1 | 19.68 | 0.7813 |
| RNN | 2 | 24.10 | 0.7693 |
| LSTM | 1 | 27.86 | 0.7785 |
| LSTM | 2 | 36.99 | 0.7900 |

### Config B — 8 partitions, 2 executors × 4 cores (15g)

| Model | Layers | Train Time (s) | Accuracy |
|---|---|---|---|
| MLP | 1 | 24.01 | 0.7863 |
| MLP | 2 | 20.44 | 0.7835 |
| MLP | 3 | 22.37 | 0.7654 |
| RNN | 1 | 22.98 | 0.7812 |
| RNN | 2 | 26.43 | 0.7835 |
| LSTM | 1 | 29.94 | 0.7886 |
| LSTM | 2 | 36.88 | 0.7752 |

### Config C — 8 partitions, 4 executors × 2 cores (7g)

| Model | Layers | Train Time (s) | Accuracy |
|---|---|---|---|
| MLP | 1 | 16.86 | 0.7917 |
| MLP | 2 | 17.18 | 0.7897 |
| MLP | 3 | 18.15 | 0.7791 |
| RNN | 1 | 18.65 | 0.7826 |
| RNN | 2 | 21.41 | 0.7794 |
| LSTM | 1 | 23.93 | 0.7834 |
| LSTM | 2 | 31.21 | 0.7779 |

---

## Observations

- **Best accuracy overall:** MLP 1-layer, Config A (0.7995) — simplest model on smallest partition config
- **Deeper layers hurt accuracy:** across all models and configs, adding layers tends to reduce accuracy slightly and always increases training time
- **Config C (more executors, less memory each) was fastest** for MLP — lower training times than Config B despite same partition count
- **LSTM is consistently slowest** to train (up to 37s) while not achieving meaningfully higher accuracy than MLP
- **RNN accuracy degrades most with depth** — RNN 2-layer consistently underperforms RNN 1-layer across all configs
- **Spark vs no-Spark:** best Spark MLP (0.7995) slightly beats the standalone MLP baseline (0.8039 is actually MLP no-Spark — Spark adds overhead without accuracy gain at this dataset size)
- **Linear SVM baseline (0.7852) outperforms all Spark deep learning models** except MLP 1-layer Config A, suggesting the dataset does not strongly benefit from deep architectures
- **KNN is the fastest baseline by far** (0.07s train) at competitive accuracy (0.7660)
