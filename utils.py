import os
import json
from glob import glob

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch


def get_dataset_csv_files(data_dir="data"):
    pattern = os.path.join(data_dir, "*.csv")
    return sorted(glob(pattern))


def build_global_type_mapping(files):
    labels = set()
    for file in files:
        df = pd.read_csv(file, low_memory=False)
        df.columns = df.columns.str.strip()
        if "type" not in df.columns:
            continue
        values = df["type"].astype(str).str.strip().tolist()
        labels.update(values)

    ordered_labels = sorted(labels)
    return {label: idx for idx, label in enumerate(ordered_labels)}


def get_normal_class_index(label_mapping):
    for label, idx in label_mapping.items():
        if str(label).strip().lower() == "normal":
            return int(idx)
    return None


def save_label_mapping(output_file, label_mapping):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    index_to_label = {str(idx): label for label, idx in label_mapping.items()}
    payload = {
        "label_to_index": label_mapping,
        "index_to_label": index_to_label,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _train_val_split(X, y, val_ratio=0.2, seed=42):
    n_samples = X.shape[0]
    val_size = max(1, int(n_samples * val_ratio)) if n_samples > 1 else 0

    if val_size == 0:
        return X, y, X, y

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_samples, generator=generator)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    if train_idx.numel() == 0:
        train_idx = val_idx

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def load_dataset_data(file, label_mapping=None, val_ratio=0.2, seed=42):
    df = pd.read_csv(file, low_memory=False)

    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    drop_cols = [col for col in ["date", "label"] if col in df.columns]
    if drop_cols:
        df = df.drop(drop_cols, axis=1)

    if "type" not in df.columns:
        raise ValueError(f"Missing required 'type' column in dataset: {file}")

    if "time" in df.columns:
        time_series = pd.to_datetime(df["time"], format="%H:%M:%S", errors="coerce")
        df["time"] = time_series.dt.hour * 60 + time_series.dt.minute

    df = df.fillna(0)

    # Normalize all feature columns into numeric values.
    # - Numeric-like columns are converted directly.
    # - Mixed/string columns are encoded after string normalization.
    for col in df.columns:
        if col == "type":
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = df[col].astype(str).str.strip()
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col])

    if label_mapping is None:
        type_encoder = LabelEncoder()
        df["type"] = type_encoder.fit_transform(df["type"].astype(str).str.strip())
        num_classes = len(type_encoder.classes_)
    else:
        df["type"] = df["type"].astype(str).str.strip().map(label_mapping)
        if df["type"].isna().any():
            raise ValueError(f"Encountered unknown label in dataset: {file}")
        df["type"] = df["type"].astype(int)
        num_classes = len(label_mapping)

    X = df.drop("type", axis=1)
    y = df["type"]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.long)
    X_train, y_train, X_val, y_val = _train_val_split(
        X_tensor,
        y_tensor,
        val_ratio=val_ratio,
        seed=seed,
    )

    return X_train, y_train, X_val, y_val, num_classes


def load_fridge_data(file, label_mapping=None, val_ratio=0.2, seed=42):
    return load_dataset_data(file, label_mapping=label_mapping, val_ratio=val_ratio, seed=seed)


def load_garage_data(file, label_mapping=None, val_ratio=0.2, seed=42):
    return load_dataset_data(file, label_mapping=label_mapping, val_ratio=val_ratio, seed=seed)
