import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch


# -------------------- Fridge --------------------

def load_fridge_data(file):

    df = pd.read_csv(file, low_memory=False)

    # Remove spaces
    df.columns = df.columns.str.strip()

    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip()

    # Drop columns
    df = df.drop(['date', 'label'], axis=1)

    # Convert time
    df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S', errors='coerce')
    df['time'] = df['time'].dt.hour * 60 + df['time'].dt.minute

    # Fill missing
    df = df.fillna(0)

    # Encode categorical
    le1 = LabelEncoder()
    le2 = LabelEncoder()

    df['temp_condition'] = le1.fit_transform(df['temp_condition'])
    df['type'] = le2.fit_transform(df['type'])

    # Split
    X = df.drop('type', axis=1)
    y = df['type']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y.values, dtype=torch.long)


# -------------------- Garage --------------------

def load_garage_data(file):

    df = pd.read_csv(file, low_memory=False)

    # Remove spaces
    df.columns = df.columns.str.strip()

    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip()

    # Drop columns
    df = df.drop(['date', 'label'], axis=1)

    # Convert time
    df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S', errors='coerce')
    df['time'] = df['time'].dt.hour * 60 + df['time'].dt.minute

    # Fill missing
    df = df.fillna(0)

    # Encode categorical
    le1 = LabelEncoder()
    le2 = LabelEncoder()
    le3 = LabelEncoder()

    df['door_state'] = le1.fit_transform(df['door_state'])
    df['sphone_signal'] = le2.fit_transform(df['sphone_signal'])
    df['type'] = le3.fit_transform(df['type'])

    # Split
    X = df.drop('type', axis=1)
    y = df['type']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y.values, dtype=torch.long)