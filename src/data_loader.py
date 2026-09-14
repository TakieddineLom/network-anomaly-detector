"""
Loads network traffic data for the anomaly detector.

Supports:
  1. NSL-KDD (KDDTrain+.txt / .arff / .csv, and KDDTest+ variants) -
     the recommended real dataset. Handles both the 43-column format
     (label = specific attack name + difficulty) and the 42-column
     format (class = normal/anomaly only), including raw .arff files.
  2. generate_synthetic_data() - a fallback so the pipeline always runs.
"""

import os
from io import StringIO
import numpy as np
import pandas as pd

NSL_KDD_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "KDDTrain+.txt")
LABEL_COLUMN = "is_anomaly"       # 0 = normal, 1 = known anomaly
ATTACK_TYPE_COLUMN = "attack_type"  # original attack name, e.g. 'neptune', 'normal'

NSL_KDD_FEATURE_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]
NSL_KDD_COLUMNS_43 = NSL_KDD_FEATURE_COLUMNS + ["label", "difficulty"]
NSL_KDD_COLUMNS_42 = NSL_KDD_FEATURE_COLUMNS + ["class"]

NUMERIC_FEATURE_COLUMNS = [
    "duration", "src_bytes", "dst_bytes", "num_failed_logins", "count",
    "srv_count", "same_srv_rate", "diff_srv_rate", "dst_host_count",
    "dst_host_srv_count",
]

# Maps specific NSL-KDD attack names to the 4 broad categories used
# in the official documentation, for readable breakdowns.
ATTACK_CATEGORY_MAP = {
    "normal": "normal",
    # DoS
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "mailbomb": "DoS", "apache2": "DoS",
    "processtable": "DoS", "udpstorm": "DoS", "worm": "DoS",
    # Probe
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "satan": "Probe", "mscan": "Probe", "saint": "Probe",
    # R2L (remote to local)
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L",
    "multihop": "R2L", "phf": "R2L", "spy": "R2L", "warezclient": "R2L",
    "warezmaster": "R2L", "sendmail": "R2L", "named": "R2L",
    "snmpgetattack": "R2L", "snmpguess": "R2L", "xlock": "R2L",
    "xsnoop": "R2L", "httptunnel": "R2L",
    # U2R (user to root)
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "ps": "U2R", "sqlattack": "U2R", "xterm": "U2R",
}


def generate_synthetic_data(n_normal: int = 2000, n_anomalies: int = 60, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    protocols = ["tcp", "udp", "icmp"]

    normal = pd.DataFrame({
        "duration": rng.normal(30, 8, n_normal).clip(1, None),
        "src_bytes": rng.normal(500, 120, n_normal).clip(0, None),
        "dst_bytes": rng.normal(450, 100, n_normal).clip(0, None),
        "num_failed_logins": rng.integers(0, 2, n_normal),
        "count": rng.normal(5, 2, n_normal).clip(1, None),
        "srv_count": rng.normal(5, 2, n_normal).clip(1, None),
        "same_srv_rate": rng.uniform(0.7, 1.0, n_normal),
        "diff_srv_rate": rng.uniform(0.0, 0.1, n_normal),
        "dst_host_count": rng.normal(150, 40, n_normal).clip(1, None),
        "dst_host_srv_count": rng.normal(140, 40, n_normal).clip(1, None),
        "protocol_type": rng.choice(protocols, n_normal, p=[0.7, 0.25, 0.05]),
    })
    normal[LABEL_COLUMN] = 0
    normal[ATTACK_TYPE_COLUMN] = "normal"

    half = n_anomalies // 2
    port_scan = pd.DataFrame({
        "duration": rng.normal(3, 1, half).clip(0.1, None),
        "src_bytes": rng.normal(80, 30, half).clip(0, None),
        "dst_bytes": rng.normal(60, 20, half).clip(0, None),
        "num_failed_logins": rng.integers(0, 1, half),
        "count": rng.normal(80, 20, half).clip(1, None),
        "srv_count": rng.normal(60, 15, half).clip(1, None),
        "same_srv_rate": rng.uniform(0.0, 0.3, half),
        "diff_srv_rate": rng.uniform(0.6, 1.0, half),
        "dst_host_count": rng.normal(255, 10, half).clip(1, 255),
        "dst_host_srv_count": rng.normal(20, 10, half).clip(1, None),
        "protocol_type": rng.choice(protocols, half, p=[0.8, 0.15, 0.05]),
    })
    port_scan[ATTACK_TYPE_COLUMN] = "portsweep"
    brute_force = pd.DataFrame({
        "duration": rng.normal(60, 15, n_anomalies - half).clip(1, None),
        "src_bytes": rng.normal(300, 80, n_anomalies - half).clip(0, None),
        "dst_bytes": rng.normal(50, 20, n_anomalies - half).clip(0, None),
        "num_failed_logins": rng.integers(15, 40, n_anomalies - half),
        "count": rng.normal(10, 3, n_anomalies - half).clip(1, None),
        "srv_count": rng.normal(10, 3, n_anomalies - half).clip(1, None),
        "same_srv_rate": rng.uniform(0.8, 1.0, n_anomalies - half),
        "diff_srv_rate": rng.uniform(0.0, 0.1, n_anomalies - half),
        "dst_host_count": rng.normal(150, 40, n_anomalies - half).clip(1, None),
        "dst_host_srv_count": rng.normal(140, 40, n_anomalies - half).clip(1, None),
        "protocol_type": rng.choice(protocols, n_anomalies - half, p=[0.9, 0.1, 0.0]),
    })
    brute_force[ATTACK_TYPE_COLUMN] = "guess_passwd"
    anomalies = pd.concat([port_scan, brute_force], ignore_index=True)
    anomalies[LABEL_COLUMN] = 1

    data = pd.concat([normal, anomalies], ignore_index=True)
    return data.sample(frac=1, random_state=seed).reset_index(drop=True)


def _encode_protocol(df: pd.DataFrame) -> pd.DataFrame:
    dummies = pd.get_dummies(df["protocol_type"], prefix="proto")
    return pd.concat([df, dummies], axis=1)


def load_nsl_kdd(path: str) -> pd.DataFrame:
    with open(path, "r", errors="ignore") as f:
        lines = f.readlines()

    data_start = 0
    for i, line in enumerate(lines):
        if line.strip().lower() == "@data":
            data_start = i + 1
            break
    data_lines = lines[data_start:]

    raw = pd.read_csv(StringIO("".join(data_lines)), header=None)
    n_cols = raw.shape[1]
    if n_cols == 43:
        raw.columns = NSL_KDD_COLUMNS_43
        attack_series = raw["label"].astype(str).str.strip().str.lower()
    elif n_cols == 42:
        raw.columns = NSL_KDD_COLUMNS_42
        attack_series = raw["class"].astype(str).str.strip().str.lower()
    else:
        raise ValueError(f"Unexpected NSL-KDD column count: {n_cols} (expected 42 or 43) in {path}")

    raw[ATTACK_TYPE_COLUMN] = attack_series
    raw[LABEL_COLUMN] = (attack_series != "normal").astype(int)
    keep = NUMERIC_FEATURE_COLUMNS + ["protocol_type", LABEL_COLUMN, ATTACK_TYPE_COLUMN]
    return raw[keep]


def _find_nsl_kdd_file():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    candidates = [
        "KDDTrain+.txt", "KDDTrain+.csv", "KDDTrain+.arff",
        "KDDTrain+_20Percent.txt", "KDDTrain+_20Percent.arff",
        "nsl_kdd_dataset.csv", "NSL_KDD_Train.csv", "Train.txt", "Train.arff",
    ]
    for name in candidates:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            return path
    return NSL_KDD_PATH


def _find_nsl_kdd_test_file():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    candidates = [
        "KDDTest+.txt", "KDDTest+.csv", "KDDTest+.arff",
        "KDDTest-21.txt", "KDDTest-21.arff",
        "nsl_kdd_test.csv", "NSL_KDD_Test.csv", "Test.txt", "Test.arff",
    ]
    for name in candidates:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            return path
    return None


def load_test_data(train_feature_columns: list):
    path = _find_nsl_kdd_test_file()
    if path is None:
        return None
    print(f"Loading real TEST data from {path}")
    df = load_nsl_kdd(path)
    df = _encode_protocol(df)
    for col in train_feature_columns:
        if col not in df.columns:
            df[col] = 0
    return df[train_feature_columns + [LABEL_COLUMN, ATTACK_TYPE_COLUMN]]


def load_data() -> pd.DataFrame:
    found_path = _find_nsl_kdd_file()
    if os.path.exists(found_path):
        print(f"Loading real data from {found_path}")
        df = load_nsl_kdd(found_path)
    else:
        print("No dataset found in data/ — using synthetic data instead.")
        df = generate_synthetic_data()
    return _encode_protocol(df)


def get_feature_columns(df: pd.DataFrame) -> list:
    proto_cols = [c for c in df.columns if c.startswith("proto_")]
    return NUMERIC_FEATURE_COLUMNS + proto_cols


def get_attack_category(attack_name: str) -> str:
    """Maps a specific attack name to its broad category (DoS/Probe/R2L/U2R/normal/Unknown)."""
    return ATTACK_CATEGORY_MAP.get(str(attack_name).strip().lower(), "Unknown")


if __name__ == "__main__":
    df = load_data()
    feature_cols = get_feature_columns(df)
    print(df.head())
    print(f"\nLoaded {len(df)} rows.")
    print(f"Feature columns: {feature_cols}")
    print(df[LABEL_COLUMN].value_counts())
