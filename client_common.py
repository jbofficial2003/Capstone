import csv
import json
import os
import logging
from pathlib import Path

import flwr as fl
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from model import SharedModel
from utils import (
    build_global_type_mapping,
    get_dataset_csv_files,
    get_normal_class_index,
    load_dataset_data,
    save_label_mapping,
)


RESULTS_DIR = "results"


class _SuppressFlowerDeprecations(logging.Filter):
    def filter(self, record):
        return "DEPRECATED FEATURE" not in record.getMessage()


logging.getLogger("flwr").addFilter(_SuppressFlowerDeprecations())


def _append_client_predictions(
    predictions_file,
    round_number,
    y_true,
    y_pred,
    index_to_label,
    normal_class_idx,
):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    file_exists = os.path.exists(predictions_file)

    with open(predictions_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "round",
                "sample_idx",
                "true_class_idx",
                "pred_class_idx",
                "true_attack_type",
                "pred_attack_type",
                "true_is_attack",
                "pred_is_attack",
            ],
        )
        if not file_exists:
            writer.writeheader()

        for sample_idx, (true_idx, pred_idx) in enumerate(zip(y_true, y_pred)):
            true_label = index_to_label.get(int(true_idx), str(true_idx))
            pred_label = index_to_label.get(int(pred_idx), str(pred_idx))

            if normal_class_idx is None:
                true_is_attack = "unknown"
                pred_is_attack = "unknown"
            else:
                true_is_attack = int(int(true_idx) != normal_class_idx)
                pred_is_attack = int(int(pred_idx) != normal_class_idx)

            writer.writerow(
                {
                    "round": round_number,
                    "sample_idx": sample_idx,
                    "true_class_idx": int(true_idx),
                    "pred_class_idx": int(pred_idx),
                    "true_attack_type": true_label,
                    "pred_attack_type": pred_label,
                    "true_is_attack": true_is_attack,
                    "pred_is_attack": pred_is_attack,
                }
            )


def _append_client_metrics(
    metrics_file,
    round_number,
    loss_value,
    accuracy,
    precision,
    recall,
    f1,
    attack_accuracy,
    attack_precision,
    attack_recall,
    attack_f1,
    cm,
):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    file_exists = os.path.exists(metrics_file)

    with open(metrics_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "round",
                "loss",
                "accuracy",
                "precision_weighted",
                "recall_weighted",
                "f1_weighted",
                "attack_accuracy",
                "attack_precision",
                "attack_recall",
                "attack_f1",
                "confusion_matrix",
            ],
        )
        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                "round": round_number,
                "loss": loss_value,
                "accuracy": accuracy,
                "precision_weighted": precision,
                "recall_weighted": recall,
                "f1_weighted": f1,
                "attack_accuracy": attack_accuracy,
                "attack_precision": attack_precision,
                "attack_recall": attack_recall,
                "attack_f1": attack_f1,
                "confusion_matrix": json.dumps(cm),
            }
        )


def run_dataset_client(dataset_file, client_name=None, server_address="localhost:8080"):
    dataset_path = Path(dataset_file)
    resolved_name = client_name or dataset_path.stem

    csv_files = get_dataset_csv_files("data")
    label_mapping = build_global_type_mapping(csv_files)
    normal_class_idx = get_normal_class_index(label_mapping)
    save_label_mapping(os.path.join(RESULTS_DIR, "label_mapping.json"), label_mapping)

    X_train, y_train, X_val, y_val, num_classes = load_dataset_data(
        str(dataset_path),
        label_mapping=label_mapping,
    )

    model = SharedModel(X_train.shape[1], num_classes)
    metrics_file = os.path.join(RESULTS_DIR, f"{resolved_name}_metrics.csv")
    predictions_file = os.path.join(RESULTS_DIR, f"{resolved_name}_predictions.csv")
    index_to_label = {idx: label for label, idx in label_mapping.items()}
    eval_state = {"round": 0}

    class DatasetClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            return [val.detach().cpu().numpy() for val in model.shared.state_dict().values()]

        def set_parameters(self, parameters):
            shared_state = model.shared.state_dict()
            if len(parameters) != len(shared_state):
                raise ValueError("Parameter length mismatch for shared layers.")

            updated_state = {}
            for key, np_value in zip(shared_state.keys(), parameters):
                updated_state[key] = torch.from_numpy(np_value).to(
                    dtype=shared_state[key].dtype,
                    device=shared_state[key].device,
                )

            model.shared.load_state_dict(updated_state, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            model.train()

            optimizer = torch.optim.Adam(model.parameters())
            loss_fn = torch.nn.CrossEntropyLoss()

            for _ in range(2):
                optimizer.zero_grad()
                output = model(X_train)
                loss = loss_fn(output, y_train)
                loss.backward()
                optimizer.step()

            return self.get_parameters(config), len(X_train), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            model.eval()
            loss_fn = torch.nn.CrossEntropyLoss()

            eval_state["round"] += 1
            round_number = int(config.get("server_round", eval_state["round"]))

            with torch.no_grad():
                output = model(X_val)
                loss = loss_fn(output, y_val)
                preds = torch.argmax(output, dim=1)
                y_true = y_val.cpu().numpy()
                y_pred = preds.cpu().numpy()

                accuracy = accuracy_score(y_true, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                )
                cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).tolist()

                if normal_class_idx is None:
                    attack_accuracy = 0.0
                    attack_precision = 0.0
                    attack_recall = 0.0
                    attack_f1 = 0.0
                else:
                    y_true_attack = (y_true != normal_class_idx).astype(int)
                    y_pred_attack = (y_pred != normal_class_idx).astype(int)
                    attack_accuracy = accuracy_score(y_true_attack, y_pred_attack)
                    attack_precision, attack_recall, attack_f1, _ = precision_recall_fscore_support(
                        y_true_attack,
                        y_pred_attack,
                        average="binary",
                        zero_division=0,
                    )

            loss_value = float(loss.detach())
            _append_client_metrics(
                metrics_file,
                round_number,
                loss_value,
                float(accuracy),
                float(precision),
                float(recall),
                float(f1),
                float(attack_accuracy),
                float(attack_precision),
                float(attack_recall),
                float(attack_f1),
                cm,
            )

            _append_client_predictions(
                predictions_file,
                round_number,
                y_true,
                y_pred,
                index_to_label,
                normal_class_idx,
            )

            return loss_value, len(X_val), {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "attack_accuracy": float(attack_accuracy),
                "attack_precision": float(attack_precision),
                "attack_recall": float(attack_recall),
                "attack_f1": float(attack_f1),
            }

    fl.client.start_numpy_client(
        server_address=server_address,
        client=DatasetClient(),
    )
