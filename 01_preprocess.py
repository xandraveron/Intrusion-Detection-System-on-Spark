# src/01_preprocess.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
import joblib

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "nsl-kdd"
ART = BASE_DIR / "artifacts"

# NSL-KDD columns (41 features + label + difficulty)
COLS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment",
    "urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted",
    "num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login",
    "is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
    "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label","difficulty"
]

# Label grouping for 5-class (NSL-KDD standard grouping)
DOS = {"back","land","neptune","pod","smurf","teardrop","mailbomb","apache2","processtable","udpstorm","worm"}
PROBE = {"satan","ipsweep","nmap","portsweep","mscan","saint"}
R2L = {"guess_passwd","ftp_write","imap","phf","multihop","warezmaster","warezclient","spy","xlock","xsnoop","snmpguess","snmpgetattack","httptunnel","sendmail","named"}
U2R = {"buffer_overflow","loadmodule","rootkit","perl","sqlattack","xterm","ps"}

def map_label(lbl: str) -> str:
    if lbl == "normal":
        return "Normal"
    if lbl in DOS:
        return "DoS"
    if lbl in PROBE:
        return "Probe"
    if lbl in R2L:
        return "R2L"
    if lbl in U2R:
        return "U2R"
    return "Other"

def load_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, names=COLS)
    df = df.drop(columns=["difficulty"])

    df["su_attempted"] = df["su_attempted"].replace(2, 0)
    df = df.drop(columns=["num_outbound_cmds"])

    for c in ["protocol_type", "service", "flag"]:
        df[c] = df[c].astype(str).str.strip()

    df["label5"] = df["label"].apply(map_label)
    df = df.drop(columns=["label"])
    return df

def main():
    train_path = DATA_DIR / "KDDTrain+.txt"
    test_path  = DATA_DIR / "KDDTest+.txt"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Missing NSL-KDD files. Expected {train_path} and {test_path}."
        )

    train_df = load_file(train_path)
    test_df  = load_file(test_path)

    y_train = train_df["label5"].values
    y_test  = test_df["label5"].values

    X_train = train_df.drop(columns=["label5"])
    X_test  = test_df.drop(columns=["label5"])

    cat_cols = ["protocol_type", "service", "flag"]
    num_cols = [c for c in X_train.columns if c not in cat_cols]


    #Min-max normalization and OneHotEncoder
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", MinMaxScaler(), num_cols),
        ],
        sparse_threshold=0,
    )

    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t  = preprocessor.transform(X_test)

    # Encode string labels to integers
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test  = le.transform(y_test)

    print("Train shape:", X_train_t.shape)
    print("Test shape:", X_test_t.shape)
    print("Final feature count:", X_train_t.shape[1])

    ART.mkdir(parents=True, exist_ok=True)

    np.save(ART / "X_train.npy", X_train_t)
    np.save(ART / "y_train.npy", y_train)
    np.save(ART / "X_test.npy", X_test_t)
    np.save(ART / "y_test.npy", y_test)

    joblib.dump(preprocessor, ART / "preprocessor.joblib")
    joblib.dump(le, ART / "label_encoder.joblib")
    print("Saved to artifacts/")

if __name__ == "__main__":
    main()
