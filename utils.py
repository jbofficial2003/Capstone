import os
import json
from glob import glob

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALLOWED_LABELS = ("backdoor", "ddos", "injection", "normal", "password")


def get_dataset_csv_files(data_dir="data"):
    pattern = os.path.join(data_dir, "*.csv")
    return sorted(glob(pattern))


def _filter_allowed_labels(series):
    allowed = {label.lower() for label in ALLOWED_LABELS}
    normalized = series.astype(str).str.strip().str.lower()
    return normalized[normalized.isin(allowed)]


def _build_timestamp(df):
    if "date" in df.columns and "time" in df.columns:
        date_part = pd.to_datetime(
            df["date"].astype(str).str.strip(),
            format="%d-%b-%y",
            errors="coerce",
        )
        time_part = pd.to_timedelta(df["time"].astype(str).str.strip(), errors="coerce")
        timestamp = date_part.dt.normalize() + time_part
    elif "date" in df.columns:
        timestamp = pd.to_datetime(
            df["date"].astype(str).str.strip(),
            format="%d-%b-%y",
            errors="coerce",
        )
    elif "time" in df.columns:
        timestamp = pd.to_datetime(
            df["time"].astype(str).str.strip(),
            format="%H:%M:%S",
            errors="coerce",
        )
    else:
        return None

    if timestamp.notna().any():
        return timestamp
    return None


def _add_lag_features(df, target_col="type", lag_steps=3):
    feature_cols = [col for col in df.columns if col != target_col]
    if not feature_cols or lag_steps < 1:
        return df

    lagged_frames = [df]
    for lag in range(1, lag_steps + 1):
        lagged = df[feature_cols].shift(lag).add_suffix(f"_lag{lag}")
        lagged_frames.append(lagged)

    return pd.concat(lagged_frames, axis=1).dropna().reset_index(drop=True)


def build_global_type_mapping(files):
    labels = set()
    for file in files:
        df = pd.read_csv(file, low_memory=False)
        df.columns = df.columns.str.strip()
        if "type" not in df.columns:
            continue
        values = _filter_allowed_labels(df["type"]).tolist()
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


def _train_val_split(X, y, val_ratio=0.2, seed=42, shuffle=True):
    n_samples = X.shape[0]
    val_size = max(1, int(n_samples * val_ratio)) if n_samples > 1 else 0

    if val_size == 0:
        return X, y, X, y

    if shuffle:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(n_samples, generator=generator)
        val_idx = indices[:val_size]
        train_idx = indices[val_size:]

        if train_idx.numel() == 0:
            train_idx = val_idx

        return X[train_idx], y[train_idx], X[val_idx], y[val_idx]

    train_size = n_samples - val_size
    if train_size <= 0:
        train_size = 1
        val_size = n_samples - train_size

    train_idx = torch.arange(0, train_size)
    val_idx = torch.arange(train_size, train_size + val_size)
    if val_idx.numel() == 0:
        val_idx = train_idx

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def load_dataset_data(file, label_mapping=None, val_ratio=0.2, seed=42):
    df = pd.read_csv(file, low_memory=False)

    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    timestamp = _build_timestamp(df)
    if timestamp is not None:
        df = df.assign(_timestamp=timestamp)
        df = df.sort_values("_timestamp", kind="mergesort").reset_index(drop=True)
        df = df.drop(columns=["_timestamp"])

    drop_cols = [col for col in ["date", "label"] if col in df.columns]
    if drop_cols:
        df = df.drop(drop_cols, axis=1)

    if "type" not in df.columns:
        raise ValueError(f"Missing required 'type' column in dataset: {file}")

    df["type"] = _filter_allowed_labels(df["type"])
    df = df.dropna(subset=["type"])

    if df.empty:
        raise ValueError(f"No rows remain after filtering to allowed labels in dataset: {file}")

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

    df = _add_lag_features(df, target_col="type", lag_steps=3)

    X = df.drop("type", axis=1)
    y = df["type"]

    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.long)

    X_train, y_train, X_val, y_val = _train_val_split(
        X_tensor,
        y_tensor,
        val_ratio=val_ratio,
        seed=seed,
        shuffle=timestamp is None,
    )

    scaler = StandardScaler()
    X_train = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
    X_val = torch.tensor(scaler.transform(X_val), dtype=torch.float32)

    return X_train, y_train, X_val, y_val, num_classes


def load_fridge_data(file, label_mapping=None, val_ratio=0.2, seed=42):
    return load_dataset_data(file, label_mapping=label_mapping, val_ratio=val_ratio, seed=seed)


def load_garage_data(file, label_mapping=None, val_ratio=0.2, seed=42):
    return load_dataset_data(file, label_mapping=label_mapping, val_ratio=val_ratio, seed=seed)
