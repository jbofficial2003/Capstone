import os
import glob
import pickle
import json
import torch
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from model import SharedModel

app = Flask(__name__, static_folder='static', static_url_path='')

RESULTS_DIR = "results"
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
LABEL_MAPPING_FILE = os.path.join(RESULTS_DIR, "label_mapping.json")

# Dynamic cache for loaded preprocessors and models
loaded_cache = {}

DEVICE_METADATA = {
    "fridge": {
        "name": "Smart Refrigerator",
        "icon": "ri-fridge-line",
        "description": "Temperature and condition log anomaly detection.",
        "feature_labels": {
            "fridge_temperature": "Fridge Temperature (°C)",
            "temp_condition": "Temperature Condition"
        },
        "categorical_choices": {
            "temp_condition": ["high", "low"]
        },
        "default_values": {
            "time": "12:00:00",
            "fridge_temperature": 2.5,
            "temp_condition": "low"
        }
    },
    "garage": {
        "name": "Garage Door Controller",
        "icon": "ri-door-open-line",
        "description": "Monitors garage door status and signal leaks.",
        "feature_labels": {
            "door_state": "Door State",
            "sphone_signal": "Phone Signal Strength"
        },
        "categorical_choices": {
            "door_state": ["closed", "open"]
        },
        "default_values": {
            "time": "12:00:00",
            "door_state": "closed",
            "sphone_signal": 0.0
        }
    },
    "gps_tracker": {
        "name": "GPS Tracker",
        "icon": "ri-map-pin-line",
        "description": "Detects spoofing or anomalies in geolocation coordinate logs.",
        "feature_labels": {
            "latitude": "Latitude",
            "longitude": "Longitude"
        },
        "categorical_choices": {},
        "default_values": {
            "time": "12:00:00",
            "latitude": 37.7749,
            "longitude": -122.4194
        }
    },
    "modbus": {
        "name": "Modbus Industrial Protocol",
        "icon": "ri-settings-5-line",
        "description": "Inspects registers for injection or password brute force attacks.",
        "feature_labels": {
            "FC1_Read_Input_Register": "FC1 Input Register",
            "FC2_Read_Discrete_Value": "FC2 Discrete Value",
            "FC3_Read_Holding_Register": "FC3 Holding Register",
            "FC4_Read_Coil": "FC4 Coil"
        },
        "categorical_choices": {},
        "default_values": {
            "time": "12:00:00",
            "FC1_Read_Input_Register": 0,
            "FC2_Read_Discrete_Value": 0,
            "FC3_Read_Holding_Register": 0,
            "FC4_Read_Coil": 0
        }
    },
    "motion_light": {
        "name": "Motion-Activated Light",
        "icon": "ri-lightbulb-line",
        "description": "Identifies lighting controller security status overrides.",
        "feature_labels": {
            "motion_status": "Motion Status (0/1)",
            "light_status": "Light Status"
        },
        "categorical_choices": {
            "light_status": ["off", "on"]
        },
        "default_values": {
            "time": "12:00:00",
            "motion_status": 0,
            "light_status": "off"
        }
    },
    "thermostat": {
        "name": "Smart Thermostat",
        "icon": "ri-temp-hot-line",
        "description": "Reviews HVAC commands and temperature settings.",
        "feature_labels": {
            "current_temperature": "Current Temperature (°C)",
            "thermostat_status": "Thermostat Status (0/1)"
        },
        "categorical_choices": {},
        "default_values": {
            "time": "12:00:00",
            "current_temperature": 20.0,
            "thermostat_status": 0
        }
    },
    "weather": {
        "name": "Weather Monitoring Station",
        "icon": "ri-cloud-windy-line",
        "description": "Analyzes weather parameters for spoofed sensor reports.",
        "feature_labels": {
            "temperature": "Temperature (°C)",
            "pressure": "Pressure (hPa)",
            "humidity": "Humidity (%)"
        },
        "categorical_choices": {},
        "default_values": {
            "time": "12:00:00",
            "temperature": 15.0,
            "pressure": 1013.25,
            "humidity": 50.0
        }
    }
}

def load_device_model(device_name):
    """Loads preprocessor and PyTorch model from disk for a given device."""
    if device_name in loaded_cache:
        return loaded_cache[device_name]

    model_path = os.path.join(MODELS_DIR, f"{device_name}_model.pth")
    prep_path = os.path.join(MODELS_DIR, f"{device_name}_preprocessor.pkl")

    if not os.path.exists(model_path) or not os.path.exists(prep_path):
        return None

    # Load preprocessor metadata
    with open(prep_path, "rb") as f:
        prep_data = pickle.load(f)

    # Reconstruct SharedModel
    input_size = prep_data["input_size"]
    num_classes = prep_data["num_classes"]
    model = SharedModel(input_size, num_classes)

    # Load state dict
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    loaded_cache[device_name] = {
        "model": model,
        "scaler": prep_data["scaler"],
        "encoders": prep_data["encoders"],
        "feature_cols": prep_data["feature_cols"],
        "num_classes": num_classes
    }
    return loaded_cache[device_name]

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/devices')
def get_devices():
    # Load label mapping if exists
    label_mapping = {}
    if os.path.exists(LABEL_MAPPING_FILE):
        try:
            with open(LABEL_MAPPING_FILE, "r") as f:
                label_mapping = json.load(f)
        except Exception:
            pass

    devices_list = []
    for key, meta in DEVICE_METADATA.items():
        # Check if model exists and is ready
        model_exists = os.path.exists(os.path.join(MODELS_DIR, f"{key}_model.pth"))
        prep_exists = os.path.exists(os.path.join(MODELS_DIR, f"{key}_preprocessor.pkl"))
        
        devices_list.append({
            "id": key,
            "name": meta["name"],
            "icon": meta["icon"],
            "description": meta["description"],
            "feature_labels": meta["feature_labels"],
            "categorical_choices": meta["categorical_choices"],
            "default_values": meta["default_values"],
            "ready": model_exists and prep_exists
        })

    return jsonify({
        "devices": devices_list,
        "label_mapping": label_mapping
    })

@app.route('/api/simulate/<device_name>')
def simulate(device_name):
    if device_name not in DEVICE_METADATA:
        return jsonify({"error": "Device not found"}), 404

    csv_path = os.path.join("data", f"{device_name}.csv")
    if not os.path.exists(csv_path):
        return jsonify({"error": f"Dataset file data/{device_name}.csv not found"}), 404

    try:
        # Load up to 100 rows to simulate streaming
        df = pd.read_csv(csv_path, low_memory=False)
        df.columns = df.columns.str.strip()
        
        # Clean data for user consumption
        df = df.fillna(0)
        # Drop columns not used as UI inputs, but keep 'type' for reference validation
        keep_cols = ["time"] + list(DEVICE_METADATA[device_name]["feature_labels"].keys()) + ["type"]
        available_cols = [col for col in keep_cols if col in df.columns]
        
        subset = df[available_cols].tail(150).to_dict(orient="records")
        return jsonify({"samples": subset})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json() or {}
    device_name = data.get("device")
    # Expects a list of 4 samples: samples[0] is current, samples[1] is lag1, samples[2] is lag2, samples[3] is lag3
    samples = data.get("samples")

    if not device_name or not samples or len(samples) < 4:
        return jsonify({"error": "Missing device or must provide exactly 4 consecutive chronological samples"}), 400

    model_data = load_device_model(device_name)
    if not model_data:
        return jsonify({"error": f"Model for device '{device_name}' is not loaded or not trained yet"}), 503

    model = model_data["model"]
    scaler = model_data["scaler"]
    encoders = model_data["encoders"]
    feature_cols = model_data["feature_cols"]

    # Extract base features (non-lag features)
    base_features = [col for col in feature_cols if "_lag" not in col]

    # Preprocess all 4 samples
    processed_samples = []
    for sample in samples:
        proc = {}
        for col in base_features:
            val = sample.get(col, 0)
            
            # Handle time conversion (HH:MM:SS to minutes)
            if col == "time":
                if isinstance(val, str):
                    try:
                        parts = val.split(':')
                        minutes = int(parts[0]) * 60 + int(parts[1])
                        val = minutes
                    except Exception:
                        val = 0
                else:
                    try:
                        val = float(val)
                    except ValueError:
                        val = 0
            # Handle categorical feature encoding
            elif col in encoders:
                val_str = str(val).strip()
                encoder = encoders[col]
                try:
                    if val_str in encoder.classes_:
                        val = int(encoder.transform([val_str])[0])
                    else:
                        # Fallback for unknown category
                        val = 0
                except Exception:
                    val = 0
            # Handle numerical values
            else:
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    val = 0.0
            proc[col] = val
        processed_samples.append(proc)

    # Reconstruct the 12-dimensional feature vector matching feature_cols order
    vector = {}
    # Current features
    for col in base_features:
        vector[col] = processed_samples[0][col]
    # Lag 1, 2, 3 features
    for lag in range(1, 4):
        for col in base_features:
            vector[f"{col}_lag{lag}"] = processed_samples[lag][col]

    try:
        # Arrange feature values into the exact expected feature sequence order
        input_list = [vector[col] for col in feature_cols]
        input_arr = np.array([input_list], dtype=np.float32)

        # Scale features
        scaled_arr = scaler.transform(input_arr)
        input_tensor = torch.tensor(scaled_arr, dtype=torch.float32)

        # Run inference
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1).numpy()[0]
            pred_class_idx = int(np.argmax(probs))

        # Retrieve label mappings
        label_mapping = {}
        if os.path.exists(LABEL_MAPPING_FILE):
            with open(LABEL_MAPPING_FILE, "r") as f:
                label_mapping = json.load(f)

        idx_to_label = label_mapping.get("index_to_label", {})
        pred_label = idx_to_label.get(str(pred_class_idx), f"Class {pred_class_idx}")
        is_attack = pred_label.lower() != "normal"

        class_probabilities = {
            idx_to_label.get(str(i), f"Class {i}"): float(probs[i])
            for i in range(len(probs))
        }

        return jsonify({
            "device": device_name,
            "prediction": pred_label,
            "class_index": pred_class_idx,
            "is_attack": is_attack,
            "probabilities": class_probabilities,
            "confidence": float(probs[pred_class_idx])
        })

    except Exception as e:
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500

if __name__ == '__main__':
    # Bind to port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
