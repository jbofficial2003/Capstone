# Federated Learning Training Pipeline

This project implements a federated learning system across multiple IoT device datasets using Flower (Federated Learning) and PyTorch.

## Quick Start

### Option 1: Automated Training (Recommended)

**Windows (PowerShell):**
```powershell
.\train.ps1
```

**Linux/macOS (Bash):**
```bash
bash train.sh
```

This will:
1. Verify dependencies
2. Start the Flower server
3. Launch all dataset clients in background processes
4. Run federated training for 5 rounds
5. Generate consolidated metrics report

### Option 2: Manual Control

Start server (Terminal 1):
```bash
python server.py
```

Start clients (Terminals 2-8):
```bash
python client_fridge.py
python client_garage.py
python client_gps_tracker.py
python client_modbus.py
python client_motion_light.py
python client_thermostat.py
python client_weather.py
```

Then generate report:
```bash
python report_metrics.py
```

## Setup

### Install Dependencies
```bash
pip install flwr torch pandas scikit-learn
```

## Architecture

### Model Structure
- **Adapter (Local):** Maps raw client features to 32-dim latent space
- **Shared Backbone (Federated):** 32→64→32 layers (aggregated across clients)
- **Personal Head (Local):** 32→16→num_classes (client-specific classifier)

Only the shared backbone is sent to server during federation. Client-specific adapters and heads remain local.

### Dataset Clients
Each dataset is a separate federated client:
- `fridge.csv` → `client_fridge.py`
- `garage.csv` → `client_garage.py`
- `gps_tracker.csv` → `client_gps_tracker.py`
- `modbus.csv` → `client_modbus.py`
- `motion_light.csv` → `client_motion_light.py`
- `thermostat.csv` → `client_thermostat.py`
- `weather.csv` → `client_weather.py`

### Time-Series Handling
- CSV rows are ordered by `date` and `time` before training.
- The loader adds lagged feature windows so each sample includes recent history from prior rows.
- Train/validation splitting is chronological when timestamps are available, which avoids leaking future rows into training.

## Output Files

After training completes, check the `results/` folder:

### Per-Client Metrics
- `fridge_metrics.csv`, `garage_metrics.csv`, etc.
- Columns: `round, loss, accuracy, precision_weighted, recall_weighted, f1_weighted, confusion_matrix`

### Server Aggregated Metrics
- `server_metrics.csv`
- Columns: `round, clients, samples, accuracy, precision, recall, f1`

### Summary Reports
- `summary_report.json` - Detailed JSON with best/last metrics per client and server
- `summary_report.csv` - Flattened CSV with all metrics for easy analysis

## Key Features

✅ Multi-dataset federated learning  
✅ Global label mapping across clients  
✅ Train/validation split per client  
✅ Per-round accuracy, precision, recall, F1  
✅ Confusion matrices for each client  
✅ Weighted aggregation on server  
✅ Persistent metric export to CSV  
✅ Automated training orchestration  
✅ Support for heterogeneous data schemas  

## Configuration

Edit these files to customize:

- `server.py` - Server strategy, number of rounds, port
- `model.py` - Network architecture
- `utils.py` - Data preprocessing, train/val split ratio
- `run_all_clients.py` - Client launch settings

## Troubleshooting

### Port already in use
Edit `server.py` and `client_*.py` files to change from `localhost:8080` to `localhost:8081` (or another port).

### Missing data file
Ensure all CSV files are in the `data/` folder with required columns:
- `date` (optional, will be dropped)
- `time` (will be converted to minutes)
- `type` (required, the label column)
- Feature columns (numeric or categorical, will be auto-encoded)

### Slow training
Reduce number of local epochs in `client_common.py` line `for _ in range(2):` to `for _ in range(1):`.

## File Structure

```
.
├── server.py                 # Flower server with weighted aggregation
├── model.py                  # SharedModel with adapter+shared+head
├── utils.py                  # Generic CSV loader and preprocessing
├── client_common.py          # Reusable client runtime
├── client_*.py               # Dataset-specific client entrypoints
├── run_all_clients.py        # Launcher for all clients
├── train.ps1                 # Windows orchestration script
├── train.sh                  # Unix orchestration script
├── report_metrics.py         # Metrics aggregation and summary generation
├── data/                     # Input CSV files (7 datasets)
└── results/                  # Output metrics CSVs and reports
```