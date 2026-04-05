import json
from pathlib import Path

import pandas as pd


RESULTS_DIR = Path("results")
SERVER_FILE = RESULTS_DIR / "server_metrics.csv"
LABEL_MAPPING_FILE = RESULTS_DIR / "label_mapping.json"
SUMMARY_JSON = RESULTS_DIR / "summary_report.json"
SUMMARY_CSV = RESULTS_DIR / "summary_report.csv"


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_server(df):
    if df.empty:
        return {"rounds": 0}

    summary = {
        "rounds": int(len(df)),
        "last_round": int(df["round"].iloc[-1]),
        "last_accuracy": safe_float(df["accuracy"].iloc[-1]),
        "last_precision": safe_float(df["precision"].iloc[-1]),
        "last_recall": safe_float(df["recall"].iloc[-1]),
        "last_f1": safe_float(df["f1"].iloc[-1]),
        "last_attack_accuracy": safe_float(df["attack_accuracy"].iloc[-1]) if "attack_accuracy" in df.columns else None,
        "last_attack_precision": safe_float(df["attack_precision"].iloc[-1]) if "attack_precision" in df.columns else None,
        "last_attack_recall": safe_float(df["attack_recall"].iloc[-1]) if "attack_recall" in df.columns else None,
        "last_attack_f1": safe_float(df["attack_f1"].iloc[-1]) if "attack_f1" in df.columns else None,
    }

    for metric in ("accuracy", "precision", "recall", "f1", "attack_accuracy", "attack_precision", "attack_recall", "attack_f1"):
        if metric not in df.columns:
            continue
        series = pd.to_numeric(df[metric], errors="coerce")
        valid = series.dropna()
        if valid.empty:
            continue
        idx = valid.idxmax()
        summary[f"best_{metric}"] = float(valid.loc[idx])
        summary[f"best_{metric}_round"] = int(df.loc[idx, "round"])

    return summary


def summarize_client(df, name):
    if df.empty:
        return {"client": name, "rounds": 0}

    summary = {
        "client": name,
        "rounds": int(len(df)),
        "last_round": int(df["round"].iloc[-1]),
        "last_loss": safe_float(df["loss"].iloc[-1]),
        "last_accuracy": safe_float(df["accuracy"].iloc[-1]),
        "last_precision": safe_float(df["precision_weighted"].iloc[-1]),
        "last_recall": safe_float(df["recall_weighted"].iloc[-1]),
        "last_f1": safe_float(df["f1_weighted"].iloc[-1]),
        "last_attack_accuracy": safe_float(df["attack_accuracy"].iloc[-1]) if "attack_accuracy" in df.columns else None,
        "last_attack_precision": safe_float(df["attack_precision"].iloc[-1]) if "attack_precision" in df.columns else None,
        "last_attack_recall": safe_float(df["attack_recall"].iloc[-1]) if "attack_recall" in df.columns else None,
        "last_attack_f1": safe_float(df["attack_f1"].iloc[-1]) if "attack_f1" in df.columns else None,
        "last_confusion_matrix": df["confusion_matrix"].iloc[-1],
    }

    metric_map = {
        "accuracy": "accuracy",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
        "f1": "f1_weighted",
        "attack_accuracy": "attack_accuracy",
        "attack_precision": "attack_precision",
        "attack_recall": "attack_recall",
        "attack_f1": "attack_f1",
    }

    for metric_name, col in metric_map.items():
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        valid = series.dropna()
        if valid.empty:
            continue
        idx = valid.idxmax()
        summary[f"best_{metric_name}"] = float(valid.loc[idx])
        summary[f"best_{metric_name}_round"] = int(df.loc[idx, "round"])

    return summary


def discover_client_metric_files():
    files = sorted(RESULTS_DIR.glob("*_metrics.csv"))
    return [f for f in files if f.name != "server_metrics.csv"]


def main():
    if not RESULTS_DIR.exists():
        raise FileNotFoundError("results folder does not exist. Run training first.")

    server_df = pd.read_csv(SERVER_FILE) if SERVER_FILE.exists() else pd.DataFrame()

    client_summaries = {}
    for file_path in discover_client_metric_files():
        client_name = file_path.stem.replace("_metrics", "")
        client_df = pd.read_csv(file_path)
        client_summaries[client_name] = summarize_client(client_df, client_name)

    summary = {
        "server": summarize_server(server_df),
        "clients": client_summaries,
    }

    if LABEL_MAPPING_FILE.exists():
        with open(LABEL_MAPPING_FILE, "r", encoding="utf-8") as f:
            summary["label_mapping"] = json.load(f)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    flat_rows = [{"scope": "server", **summary["server"]}]
    for client_name, client_summary in summary["clients"].items():
        flat_rows.append({"scope": f"client_{client_name}", **client_summary})

    pd.DataFrame(flat_rows).to_csv(SUMMARY_CSV, index=False)

    print(f"Saved JSON summary to: {SUMMARY_JSON}")
    print(f"Saved CSV summary to: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
