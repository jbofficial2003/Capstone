#!/bin/bash

# train.sh - Complete training orchestration script for federated learning
# Usage: bash train.sh

echo "================================"
echo "Federated Learning Training Hub"
echo "================================"
echo ""

# Check dependencies
echo "[CHECK] Verifying dependencies..."

required_packages=("flwr" "torch" "pandas" "sklearn")
missing=()

for package in "${required_packages[@]}"; do
    if python3 -c "import ${package%% *}" 2>/dev/null; then
        echo "[OK]    $package is installed"
    else
        missing+=("$package")
    fi
done

if [ ${#missing[@]} -gt 0 ]; then
    echo "[WARN]  Missing packages: ${missing[*]}"
    echo "[HINT]  Run: pip install flwr torch pandas scikit-learn"
    read -p "Press Enter to continue anyway, or Ctrl+C to exit"
fi

echo ""
echo "[START] Launching federated training..."
echo "[INFO]  Check console output for training progress"
echo ""

# Launch training
python3 run_all_clients.py

echo ""
echo "[COMPLETE] Training finished"
echo "[INFO]    Generating consolidated metrics report..."

# Generate report
python3 report_metrics.py

echo ""
echo "[SUCCESS] Training and reporting complete!"
echo "[OUTPUT]  Check results/ folder for detailed metrics:"
echo "          - results/server_metrics.csv"
echo "          - results/*_metrics.csv (per-client)"
echo "          - results/summary_report.json"
echo "          - results/summary_report.csv"
echo ""
