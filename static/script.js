// Global Application State
let appDevices = [];
let labelMapping = {};
let selectedDevice = null;

// Simulation State
let simulationRows = [];
let simulationIndex = 0;
let simulationIntervalId = null;
let historyBuffer = []; // Last 4 samples for lag context
let processedCount = 0;
let anomalyCount = 0;

// Chart Instance
let threatChart = null;
let threatDistributionCounts = {};

// On Page Load
window.addEventListener("DOMContentLoaded", () => {
    fetchDevices();
    setupEventListeners();
});

// Fetch Available Devices from Backend
function fetchDevices() {
    fetch("/api/devices")
        .then(res => res.json())
        .then(data => {
            appDevices = data.devices;
            labelMapping = data.label_mapping;
            renderDeviceList();
            updateGlobalStatus();
        })
        .catch(err => {
            console.error("Error fetching devices:", err);
            const container = document.getElementById("device-list-container");
            container.innerHTML = `<div class="error-text"><i class="ri-error-warning-line"></i> Error loading devices. Is the server running?</div>`;
        });
}

// Render the Sidebar Device List
function renderDeviceList() {
    const container = document.getElementById("device-list-container");
    container.innerHTML = "";

    appDevices.forEach(device => {
        const card = document.createElement("div");
        card.className = `device-card ${device.ready ? "" : "disabled"}`;
        if (selectedDevice && selectedDevice.id === device.id) {
            card.classList.add("active");
        }

        card.innerHTML = `
            <div class="d-icon">
                <i class="${device.icon || 'ri-cpu-line'}"></i>
            </div>
            <div class="d-info">
                <h3>${device.name}</h3>
                <p>${device.description}</p>
            </div>
            <div class="d-status-dot ${device.ready ? 'active' : ''}"></div>
        `;

        if (device.ready) {
            card.addEventListener("click", () => selectDevice(device));
        } else {
            card.title = "Model not trained yet. Run train.ps1 first.";
        }

        container.appendChild(card);
    });
}

// Update navbar status labels
function updateGlobalStatus() {
    const badge = document.getElementById("active-devices-badge");
    const activeCount = appDevices.filter(d => d.ready).length;
    badge.innerHTML = `<i class="ri-cpu-line"></i> <span>${activeCount}/${appDevices.length} Models Loaded</span>`;
}

// Select a Device and Open Dashboard
function selectDevice(device) {
    // Clean up current running simulation
    stopSimulation();
    
    selectedDevice = device;
    
    // Highlight selected card in sidebar
    renderDeviceList();
    
    // Switch Views
    document.getElementById("welcome-view").classList.add("hidden");
    const monitorView = document.getElementById("monitoring-view");
    monitorView.classList.remove("hidden");
    
    // Set Device Info Header
    document.getElementById("monitored-device-icon").innerHTML = `<i class="${device.icon}"></i>`;
    document.getElementById("monitored-device-name").textContent = device.name;
    document.getElementById("monitored-device-desc").textContent = device.description;

    // Reset Metrics
    resetMetrics();

    // Reset charts
    resetCharts();

    // Prepare Tabs & Inputs
    generateManualForm();

    // Fetch simulation data from csv
    fetchSimulationData(device.id);
}

// Reset metric cards
function resetMetrics() {
    processedCount = 0;
    anomalyCount = 0;
    historyBuffer = [];
    document.getElementById("processed-packets-count").textContent = "0";
    document.getElementById("anomaly-ratio").textContent = "0.0%";
    document.getElementById("anomaly-ratio").className = "m-value warning-text";
    document.getElementById("live-threat-status").innerHTML = `<i class="ri-shield-check-line"></i> Normal`;
    document.getElementById("live-threat-status").className = "m-value normal-text";
    document.getElementById("live-threat-sub").textContent = "Awaiting event packet capture...";
    document.getElementById("live-confidence").textContent = "--%";
    document.getElementById("live-confidence-bar").style.width = "0%";
    document.getElementById("live-confidence-bar").className = "progress-bar-fill green-bg";
    
    // Clear Table Log
    const tbody = document.getElementById("log-table-body");
    tbody.innerHTML = `
        <tr class="placeholder-row">
            <td colspan="6">No events captured in this session. Start the stream or inject manual vectors to display data.</td>
        </tr>
    `;
}

// Fetch Pre-processed rows from backend to feed the simulation
function fetchSimulationData(deviceId) {
    fetch(`/api/simulate/${deviceId}`)
        .then(res => res.json())
        .then(data => {
            if (data.error) {
                console.error(data.error);
                return;
            }
            simulationRows = data.samples;
            simulationIndex = 0;
            
            // Enable control buttons
            document.getElementById("btn-start-stream").disabled = false;
        })
        .catch(err => console.error("Error loading simulation CSV data:", err));
}

// Set up UI Event Listeners
function setupEventListeners() {
    // Mode switcher buttons
    const tabSimulate = document.getElementById("tab-simulate-btn");
    const tabManual = document.getElementById("tab-manual-btn");
    const paneSimulate = document.getElementById("pane-simulate");
    const paneManual = document.getElementById("pane-manual");

    tabSimulate.addEventListener("click", () => {
        tabSimulate.classList.add("active");
        tabManual.classList.remove("active");
        paneSimulate.classList.remove("hidden");
        paneManual.classList.add("hidden");
    });

    tabManual.addEventListener("click", () => {
        tabManual.classList.add("active");
        tabSimulate.classList.remove("active");
        paneManual.classList.remove("hidden");
        paneSimulate.classList.add("hidden");
    });

    // Stream Controls
    document.getElementById("btn-start-stream").addEventListener("click", startSimulation);
    document.getElementById("btn-pause-stream").addEventListener("click", stopSimulation);
    document.getElementById("btn-reset-stream").addEventListener("click", () => {
        stopSimulation();
        resetMetrics();
        resetCharts();
        simulationIndex = 0;
    });

    // Clear logs button
    document.getElementById("btn-clear-log").addEventListener("click", () => {
        const tbody = document.getElementById("log-table-body");
        tbody.innerHTML = `
            <tr class="placeholder-row">
                <td colspan="6">Log cleared. Start the stream to display new captures.</td>
            </tr>
        `;
    });

    // Anomaly slider
    const slider = document.getElementById("threshold-slider");
    const sliderVal = document.getElementById("threshold-val");
    slider.addEventListener("input", () => {
        sliderVal.textContent = `${slider.value}%`;
    });

    // Form submission
    const form = document.getElementById("manual-vector-form");
    form.addEventListener("submit", (e) => {
        e.preventDefault();
        submitManualVector();
    });
}

// Generate Manual Vector Input Form Dynamically
function generateManualForm() {
    const container = document.getElementById("dynamic-inputs-container");
    container.innerHTML = "";
    
    if (!selectedDevice) return;

    // Time Input
    const timeGroup = document.createElement("div");
    timeGroup.className = "form-group";
    timeGroup.innerHTML = `
        <label for="input-time">Packet Timestamp (HH:MM:SS)</label>
        <input type="text" id="input-time" class="form-control" value="12:00:00" placeholder="e.g. 14:32:05">
    `;
    container.appendChild(timeGroup);

    // Dynamic Features
    Object.entries(selectedDevice.feature_labels).forEach(([key, label]) => {
        const group = document.createElement("div");
        group.className = "form-group";

        const isCategorical = selectedDevice.categorical_choices[key] !== undefined;
        
        if (isCategorical) {
            const choices = selectedDevice.categorical_choices[key];
            let options = choices.map(c => `<option value="${c}">${c}</option>`).join("");
            group.innerHTML = `
                <label for="input-${key}">${label}</label>
                <select id="input-${key}" class="form-control">
                    ${options}
                </select>
            `;
        } else {
            const defaultVal = selectedDevice.default_values[key] || 0.0;
            group.innerHTML = `
                <label for="input-${key}">${label}</label>
                <input type="number" step="any" id="input-${key}" class="form-control" value="${defaultVal}">
            `;
        }
        container.appendChild(group);
    });
}

// Start Stream Simulation
function startSimulation() {
    if (simulationIntervalId) return;

    document.getElementById("btn-start-stream").disabled = true;
    document.getElementById("btn-pause-stream").disabled = false;

    const speedSelect = document.getElementById("stream-speed");
    let speed = parseInt(speedSelect.value);

    // Watch for speed updates
    speedSelect.addEventListener("change", () => {
        if (simulationIntervalId) {
            stopSimulation();
            startSimulation();
        }
    });

    simulationIntervalId = setInterval(() => {
        if (simulationIndex >= simulationRows.length) {
            stopSimulation();
            alert("End of dataset simulation. Restarting from index 0.");
            simulationIndex = 0;
            return;
        }

        const rawRow = simulationRows[simulationIndex++];
        processSimulatedRow(rawRow);
    }, speed);
}

// Stop Stream Simulation
function stopSimulation() {
    if (simulationIntervalId) {
        clearInterval(simulationIntervalId);
        simulationIntervalId = null;
    }
    document.getElementById("btn-start-stream").disabled = false;
    document.getElementById("btn-pause-stream").disabled = true;
}

// Process a Streamed Row
function processSimulatedRow(row) {
    // Add row to our rolling history buffer
    historyBuffer.unshift(row);
    if (historyBuffer.length > 4) {
        historyBuffer.pop();
    }

    // Build the lag samples array
    let samples = [];
    for (let i = 0; i < 4; i++) {
        if (i < historyBuffer.length) {
            samples.push(historyBuffer[i]);
        } else {
            // Pad with the oldest item
            samples.push(historyBuffer[historyBuffer.length - 1]);
        }
    }

    // Send prediction request to Flask server
    fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            device: selectedDevice.id,
            samples: samples
        })
    })
    .then(res => {
        if (!res.ok) throw new Error("Server prediction error");
        return res.json();
    })
    .then(predictionResult => {
        updateUIWithPrediction(row, predictionResult);
    })
    .catch(err => {
        console.error("Prediction error:", err);
    });
}

// Submit Manual Feature Input to Server
function submitManualVector() {
    if (!selectedDevice) return;

    // Build current sample
    const currentSample = {
        time: document.getElementById("input-time").value
    };

    Object.keys(selectedDevice.feature_labels).forEach(key => {
        const element = document.getElementById(`input-${key}`);
        let val = element.value;
        if (element.type === "number") {
            val = parseFloat(val);
        }
        currentSample[key] = val;
    });

    // For manual vector, we don't have true stream history, so we fill the lag states 
    // with copies of this manual entry to establish the 4-step sequence
    const samples = [currentSample, currentSample, currentSample, currentSample];

    fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            device: selectedDevice.id,
            samples: samples
        })
    })
    .then(res => res.json())
    .then(predictionResult => {
        if (predictionResult.error) {
            alert("Prediction failed: " + predictionResult.error);
            return;
        }
        // Manual entry has no ground truth target label
        const rowWithNoTruth = { ...currentSample, type: "Unknown" };
        updateUIWithPrediction(rowWithNoTruth, predictionResult);
    })
    .catch(err => {
        console.error("Prediction error:", err);
    });
}

// Update Stats Cards and Event Log Table
function updateUIWithPrediction(rawRow, result) {
    processedCount++;
    document.getElementById("processed-packets-count").textContent = processedCount;

    // Apply custom slider threshold for attacks if the user has changed it.
    // If logits predict attack with probability >= threshold, alert it.
    // Result has "probabilities" dictionary. We find which ones are attacks.
    const isAttackPredicted = checkSliderThresholdOverride(result);

    if (isAttackPredicted) {
        anomalyCount++;
    }

    // Threat percentage
    const hitRate = ((anomalyCount / processedCount) * 100).toFixed(1);
    document.getElementById("anomaly-ratio").textContent = `${hitRate}%`;
    document.getElementById("anomaly-ratio").className = `m-value ${anomalyCount > 0 ? "danger-text" : "warning-text"}`;

    // Confidence
    const confidenceText = (result.confidence * 100).toFixed(1) + "%";
    document.getElementById("live-confidence").textContent = confidenceText;
    const confBar = document.getElementById("live-confidence-bar");
    confBar.style.width = confidenceText;

    // Global threat card
    const threatCard = document.getElementById("live-threat-status");
    const threatSub = document.getElementById("live-threat-sub");

    if (isAttackPredicted) {
        threatCard.innerHTML = `<i class="ri-error-warning-line"></i> Attack Detected`;
        threatCard.className = "m-value danger-text";
        threatSub.textContent = `Alert: ${result.prediction.toUpperCase()} activity intercepted!`;
        confBar.className = "progress-bar-fill red-bg";
    } else {
        threatCard.innerHTML = `<i class="ri-shield-check-line"></i> Normal`;
        threatCard.className = "m-value normal-text";
        threatSub.textContent = `System operating within safe bounds.`;
        confBar.className = "progress-bar-fill green-bg";
    }

    // Add to logs table
    appendRowToLogTable(rawRow, result, isAttackPredicted);

    // Update charts data distribution
    updateThreatDistributionChart(result.prediction);
}

// Check if slider threshold overrides classification label
function checkSliderThresholdOverride(result) {
    const thresholdSlider = document.getElementById("threshold-slider");
    const thresholdFraction = parseFloat(thresholdSlider.value) / 100.0;
    
    // Find the attack class index or name
    let attackProbabilitySum = 0;
    Object.entries(result.probabilities).forEach(([className, prob]) => {
        if (className.toLowerCase() !== "normal") {
            attackProbabilitySum += prob;
        }
    });

    return attackProbabilitySum >= thresholdFraction;
}

// Append rows to UI Logger Table
function appendRowToLogTable(rawRow, result, isAttackPredicted) {
    const tbody = document.getElementById("log-table-body");
    
    // Remove placeholder row
    const placeholder = tbody.querySelector(".placeholder-row");
    if (placeholder) {
        placeholder.remove();
    }

    // Format attributes
    let attributesHtml = `<span class="attr-pill">time: ${rawRow.time}</span>`;
    Object.entries(selectedDevice.feature_labels).forEach(([key, label]) => {
        attributesHtml += ` <span class="attr-pill">${key}: ${rawRow[key]}</span>`;
    });

    const tr = document.createElement("tr");
    tr.className = isAttackPredicted ? "attack-row" : "";
    
    const confidenceText = (result.confidence * 100).toFixed(1) + "%";
    const badgeClass = isAttackPredicted ? "attack" : "normal";
    const badgeText = isAttackPredicted ? `${result.prediction}` : "Normal";
    const badgeIcon = isAttackPredicted ? "ri-alert-line" : "ri-shield-check-line";

    tr.innerHTML = `
        <td style="color: var(--text-muted); font-family: monospace;">${new Date().toLocaleTimeString()}</td>
        <td><div class="attr-pills">${attributesHtml}</div></td>
        <td><span class="truth-badge">${rawRow.type || 'N/A'}</span></td>
        <td style="font-weight: 600; color: ${isAttackPredicted ? 'var(--color-danger)' : 'var(--color-success)'};">${result.prediction}</td>
        <td style="font-weight: 500;">${confidenceText}</td>
        <td><span class="lbl-badge ${badgeClass}"><i class="${badgeIcon}"></i> ${badgeText}</span></td>
    `;

    // Keep log table to max 100 rows
    if (tbody.children.length >= 100) {
        tbody.removeChild(tbody.lastChild);
    }
    // Insert at front
    tbody.insertBefore(tr, tbody.firstChild);
}

// Reset distribution counters
function resetCharts() {
    threatDistributionCounts = {};
    if (threatChart) {
        threatChart.destroy();
        threatChart = null;
    }
}

// Update the chart visual counts
function updateThreatDistributionChart(predictedLabel) {
    threatDistributionCounts[predictedLabel] = (threatDistributionCounts[predictedLabel] || 0) + 1;

    const labels = Object.keys(threatDistributionCounts);
    const data = Object.values(threatDistributionCounts);

    // Map labels to high-contrast colors
    const colors = labels.map(lbl => {
        if (lbl.toLowerCase() === "normal") return "#10b981"; // Emerald
        if (lbl.toLowerCase() === "ddos") return "#ef4444"; // Red
        if (lbl.toLowerCase() === "injection") return "#f59e0b"; // Yellow
        if (lbl.toLowerCase() === "backdoor") return "#8b5cf6"; // Purple
        if (lbl.toLowerCase() === "password") return "#ec4899"; // Pink
        return "#6366f1"; // default Indigo
    });

    if (!threatChart) {
        const ctx = document.getElementById("threat-distribution-chart").getContext("2d");
        threatChart = new Chart(ctx, {
            type: "doughnut",
            data: {
                labels: labels,
                datasets: [{
                    data: data,
                    backgroundColor: colors,
                    borderWidth: 1,
                    borderColor: "rgba(255, 255, 255, 0.08)"
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: "bottom",
                        labels: {
                            color: "#94a3b8",
                            font: { family: "Outfit", size: 11 }
                        }
                    }
                },
                cutout: "65%"
            }
        });
    } else {
        threatChart.data.labels = labels;
        threatChart.data.datasets[0].data = data;
        threatChart.data.datasets[0].backgroundColor = colors;
        threatChart.update();
    }
}
