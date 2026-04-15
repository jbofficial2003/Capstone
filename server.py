import flwr as fl
import csv
import os
import logging


round_state = {"eval_round": 0}
RESULTS_DIR = "results"
SERVER_METRICS_FILE = os.path.join(RESULTS_DIR, "server_metrics.csv")
METRIC_KEYS = ("accuracy", "precision", "recall", "f1")
ATTACK_METRIC_KEYS = ("attack_accuracy", "attack_precision", "attack_recall", "attack_f1")


class _SuppressFlowerDeprecations(logging.Filter):
    def filter(self, record):
        return "DEPRECATED FEATURE" not in record.getMessage()


logging.getLogger("flwr").addFilter(_SuppressFlowerDeprecations())


def append_server_metrics(round_number, client_count, sample_count, aggregated_metrics):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    file_exists = os.path.exists(SERVER_METRICS_FILE)

    with open(SERVER_METRICS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "round",
                "clients",
                "samples",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "attack_accuracy",
                "attack_precision",
                "attack_recall",
                "attack_f1",
            ],
        )
        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                "round": round_number,
                "clients": client_count,
                "samples": sample_count,
                "accuracy": aggregated_metrics.get("accuracy"),
                "precision": aggregated_metrics.get("precision"),
                "recall": aggregated_metrics.get("recall"),
                "f1": aggregated_metrics.get("f1"),
                "attack_accuracy": aggregated_metrics.get("attack_accuracy"),
                "attack_precision": aggregated_metrics.get("attack_precision"),
                "attack_recall": aggregated_metrics.get("attack_recall"),
                "attack_f1": aggregated_metrics.get("attack_f1"),
            }
        )


def _aggregate_metrics(metrics):
    all_metric_keys = METRIC_KEYS + ATTACK_METRIC_KEYS
    weighted_sums = {k: 0.0 for k in all_metric_keys}
    weighted_counts = {k: 0 for k in all_metric_keys}
    total_examples = sum(num_examples for num_examples, _ in metrics)

    for num_examples, metric_dict in metrics:
        for key in all_metric_keys:
            value = metric_dict.get(key)
            if value is None:
                continue
            weighted_sums[key] += num_examples * float(value)
            weighted_counts[key] += num_examples

    aggregated_metrics = {
        key: (weighted_sums[key] / weighted_counts[key])
        for key in all_metric_keys
        if weighted_counts[key] > 0
    }

    return aggregated_metrics, total_examples


def aggregate_fit_metrics(metrics):
    aggregated_metrics, total_examples = _aggregate_metrics(metrics)

    if not aggregated_metrics:
        print("[FIT] metrics=N/A (no client metrics)")
        return {"accuracy": 0.0}

    parts = [
        f"{key}={aggregated_metrics[key]:.4f}"
        for key in METRIC_KEYS
        if key in aggregated_metrics
    ]
    print(f"[FIT] {' | '.join(parts)} | clients={len(metrics)} | samples={total_examples}")
    return aggregated_metrics


def aggregate_eval_metrics(metrics):
    round_state["eval_round"] += 1
    round_number = round_state["eval_round"]

    aggregated_metrics, total_examples = _aggregate_metrics(metrics)

    if not aggregated_metrics:
        print(f"[Round {round_number}] eval_metrics=N/A (no client metrics)")
        append_server_metrics(round_number, len(metrics), total_examples, {"accuracy": 0.0})
        return {"accuracy": 0.0}

    parts = [
        f"{key}={aggregated_metrics[key]:.4f}"
        for key in (METRIC_KEYS + ATTACK_METRIC_KEYS)
        if key in aggregated_metrics
    ]
    print(
        f"[Round {round_number}] {' | '.join(parts)} | clients={len(metrics)} | samples={total_examples}"
    )

    append_server_metrics(round_number, len(metrics), total_examples, aggregated_metrics)
    return aggregated_metrics


strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=7,
    min_evaluate_clients=7,
    min_available_clients=7,
    evaluate_metrics_aggregation_fn=aggregate_eval_metrics,
    fit_metrics_aggregation_fn=aggregate_fit_metrics,
)

fl.server.start_server(
    server_address="localhost:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
)