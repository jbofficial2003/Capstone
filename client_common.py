import csv
import json
import os
import logging
from pathlib import Path

import flwr as fl
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from model import SharedModel
from utils import (
    build_global_type_mapping,
    get_dataset_csv_files,
    get_normal_class_index,
    load_dataset_data,
    save_label_mapping,
)


RESULTS_DIR = "results"
LOCAL_EPOCHS = 12
LOCAL_LR = 2e-3
TARGET_ACCURACY_FLOOR = 0.90
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GLOBAL_BLEND_DEFAULT = 0.10
HARD_CLIENT_BLEND = {
    "fridge": 0.0,
    "garage": 0.0,
    "modbus": 0.0,
    "motion_light": 0.0,
}


def _predict_with_threshold(logits, normal_class_idx, threshold):
    if normal_class_idx is None or logits.shape[1] != 2:
        return torch.argmax(logits, dim=1)

    attack_class_idx = 1 - int(normal_class_idx)
    probs = torch.softmax(logits, dim=1)
    attack_probs = probs[:, attack_class_idx]

    preds = torch.full_like(torch.argmax(logits, dim=1), int(normal_class_idx))
    preds = torch.where(
        attack_probs >= float(threshold),
        torch.full_like(preds, attack_class_idx),
        preds,
    )
    return preds


def _best_threshold_for_recall(logits, y_true, normal_class_idx, min_accuracy):
    if normal_class_idx is None or logits.shape[1] != 2:
        return 0.5

    attack_class_idx = 1 - int(normal_class_idx)
    attack_probs = torch.softmax(logits, dim=1)[:, attack_class_idx].detach().cpu().numpy()
    y_true_np = y_true.detach().cpu().numpy()
    y_true_attack = (y_true_np != int(normal_class_idx)).astype(int)

    thresholds = np.linspace(0.05, 0.95, 19)
    best_threshold = 0.5
    best_recall = -1.0
    best_accuracy = -1.0

    fallback_threshold = 0.5
    fallback_accuracy = -1.0
    fallback_recall = -1.0

    for threshold in thresholds:
        y_pred_attack = (attack_probs >= threshold).astype(int)
        accuracy = accuracy_score(y_true_attack, y_pred_attack)
        _, recall, _, _ = precision_recall_fscore_support(
            y_true_attack,
            y_pred_attack,
            average="binary",
            zero_division=0,
        )

        if (
            accuracy > fallback_accuracy
            or (accuracy == fallback_accuracy and recall > fallback_recall)
        ):
            fallback_accuracy = float(accuracy)
            fallback_recall = float(recall)
            fallback_threshold = float(threshold)

        if accuracy >= min_accuracy and (
            recall > best_recall
            or (recall == best_recall and accuracy > best_accuracy)
        ):
            best_recall = float(recall)
            best_accuracy = float(accuracy)
            best_threshold = float(threshold)

    if best_recall >= 0.0:
        return best_threshold
    return fallback_threshold


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

    model = SharedModel(X_train.shape[1], num_classes).to(DEVICE)
    metrics_file = os.path.join(RESULTS_DIR, f"{resolved_name}_metrics.csv")
    predictions_file = os.path.join(RESULTS_DIR, f"{resolved_name}_predictions.csv")
    index_to_label = {idx: label for label, idx in label_mapping.items()}
    eval_state = {"round": 0}
    client_blend = float(HARD_CLIENT_BLEND.get(resolved_name, GLOBAL_BLEND_DEFAULT))

    X_val = X_val.to(DEVICE)
    y_val = y_val.to(DEVICE)

    train_dataset = TensorDataset(X_train, y_train)
    class_counts = torch.bincount(y_train, minlength=num_classes).float()
    class_counts = torch.clamp(class_counts, min=1.0)
    weights = 1.0 / torch.sqrt(class_counts)
    weights = weights / weights.sum() * num_classes
    sample_weights = weights[y_train].double()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=min(256, len(train_dataset)),
        shuffle=False,
        sampler=sampler,
        pin_memory=torch.cuda.is_available(),
    )

    class DatasetClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            return [val.detach().cpu().numpy() for val in model.shared.state_dict().values()]

        def set_parameters(self, parameters):
            shared_state = model.shared.state_dict()
            if len(parameters) != len(shared_state):
                raise ValueError("Parameter length mismatch for shared layers.")

            updated_state = {}
            for key, np_value in zip(shared_state.keys(), parameters):
                global_tensor = torch.from_numpy(np_value).to(
                    dtype=shared_state[key].dtype,
                    device=shared_state[key].device,
                )
                local_tensor = shared_state[key]
                updated_state[key] = (
                    client_blend * global_tensor + (1.0 - client_blend) * local_tensor
                )

            model.shared.load_state_dict(updated_state, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            model.train()

            optimizer = torch.optim.AdamW(model.parameters(), lr=LOCAL_LR, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=LOCAL_EPOCHS,
                eta_min=LOCAL_LR * 0.2,
            )
            loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.02)
            best_state = None
            best_score = float("-inf")

            def _model_selection_score():
                model.eval()
                with torch.no_grad():
                    val_logits = model(X_val)
                    val_preds = torch.argmax(val_logits, dim=1).cpu().numpy()
                    val_true = y_val.cpu().numpy()

                    accuracy = accuracy_score(val_true, val_preds)
                    score = float(accuracy)

                model.train()
                return score

            for _ in range(LOCAL_EPOCHS):
                for xb, yb in train_loader:
                    xb = xb.to(DEVICE, non_blocking=torch.cuda.is_available())
                    yb = yb.to(DEVICE, non_blocking=torch.cuda.is_available())
                    optimizer.zero_grad()
                    output = model(xb)
                    loss = loss_fn(output, yb)
                    loss.backward()
                    optimizer.step()

                epoch_score = _model_selection_score()
                if epoch_score > best_score:
                    best_score = epoch_score
                    best_state = {
                        key: value.detach().clone()
                        for key, value in model.state_dict().items()
                    }

                scheduler.step()

            if best_state is not None:
                model.load_state_dict(best_state, strict=True)

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
