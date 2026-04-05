import subprocess
import sys
import time
from pathlib import Path

DATASETS = [
    "fridge",
    "garage",
    "gps_tracker",
    "modbus",
    "motion_light",
    "thermostat",
    "weather",
]

SERVER_SCRIPT = "server.py"
CLIENT_TEMPLATE = "client_{}.py"
RESULTS_DIR = Path("results")


def launch_process(script_name, name):
    print(f"[LAUNCHER] Starting {name}...")
    try:
        # Inherit console stdio to avoid deadlocks from unconsumed PIPE buffers.
        process = subprocess.Popen([sys.executable, script_name])
        return process
    except Exception as e:
        print(f"[ERROR] Failed to start {name}: {e}")
        return None


def main():
    print("[LAUNCHER] Starting Federated Learning Training")
    print(f"[LAUNCHER] Python executable: {sys.executable}")
    print(f"[LAUNCHER] Datasets: {', '.join(DATASETS)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    server_proc = launch_process(SERVER_SCRIPT, "SERVER")
    if not server_proc:
        print("[ERROR] Failed to start server. Aborting.")
        sys.exit(1)

    time.sleep(2)

    client_procs = {}
    for dataset in DATASETS:
        client_script = CLIENT_TEMPLATE.format(dataset)
        if not Path(client_script).exists():
            print(f"[WARN] Client script not found: {client_script}")
            continue

        proc = launch_process(client_script, f"CLIENT_{dataset.upper()}")
        if proc:
            client_procs[dataset] = proc
            time.sleep(0.5)

    print(f"[LAUNCHER] Started server and {len(client_procs)} clients")

    try:
        server_code = server_proc.wait()
        print(f"[LAUNCHER] Server exited with code {server_code}")
    except KeyboardInterrupt:
        print("\n[LAUNCHER] Interrupt received. Stopping processes...")
        server_proc.terminate()
        for dataset, proc in client_procs.items():
            print(f"[LAUNCHER] Stopping CLIENT_{dataset.upper()}...")
            proc.terminate()

        time.sleep(1)
        for dataset, proc in client_procs.items():
            if proc.poll() is None:
                proc.kill()

        if server_proc.poll() is None:
            server_proc.kill()
        print("[LAUNCHER] All processes stopped.")

    failed_clients = [name for name, proc in client_procs.items() if proc.returncode not in (0, None)]
    if failed_clients:
        print(f"[WARN] Some clients exited with non-zero code: {', '.join(failed_clients)}")


if __name__ == "__main__":
    main()
