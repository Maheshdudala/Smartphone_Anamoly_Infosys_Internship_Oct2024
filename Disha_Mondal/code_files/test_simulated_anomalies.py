import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Original Data Setup
data = augmented_data.copy()
features = ['Acc X', 'Acc Y', 'Acc Z', 'gyro_x', 'gyro_y', 'gyro_z']  # Define feature columns for models
data['anomaly'] = 0  # Initialize anomaly column for simulated data

# Function to Simulate Different Types of Anomalies
def simulate_anomalies(data, feature, anomaly_type="spike", magnitude=10, frequency=0.05):
    data_sim = data.copy()
    anomaly_indices = np.random.choice(data_sim.index, int(frequency * len(data_sim)), replace=False)
    
    if anomaly_type == "spike":
        data_sim.loc[anomaly_indices, feature] += magnitude * np.random.randn(len(anomaly_indices))
    elif anomaly_type == "drift":
        data_sim.loc[anomaly_indices, feature] += np.linspace(0, magnitude, len(anomaly_indices))
    elif anomaly_type == "drop":
        data_sim.loc[anomaly_indices, feature] = data_sim[feature].min()
    elif anomaly_type == "noise":
        data_sim.loc[anomaly_indices, feature] += magnitude * np.random.uniform(-1, 1, len(anomaly_indices))
    
    data_sim.loc[anomaly_indices, 'anomaly'] = 1
    return data_sim

# Initialize results dictionary
results = {"Model": [], "Anomaly Type": [], "Accuracy": [], "Precision": [], "Recall": [], "F1 Score": []}

# Types of anomalies to test
anomaly_types = ["spike", "drift", "drop", "noise"]

# Isolation Forest and LOF Setup
iso_forest = IsolationForest(contamination=0.05, random_state=42)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True)

for anomaly_type in anomaly_types:
    # Simulate anomalies
    simulated_data = simulate_anomalies(data, feature='Acc X', anomaly_type=anomaly_type, magnitude=10, frequency=0.1)

    # Isolation Forest Model Evaluation
    iso_forest.fit(simulated_data[features])
    iso_preds = iso_forest.predict(simulated_data[features])
    iso_preds = (iso_preds == -1).astype(int)
    
    accuracy_iso = accuracy_score(simulated_data['anomaly'], iso_preds)
    precision_iso = precision_score(simulated_data['anomaly'], iso_preds)
    recall_iso = recall_score(simulated_data['anomaly'], iso_preds)
    f1_iso = f1_score(simulated_data['anomaly'], iso_preds)
    
    # Log Isolation Forest results
    results["Model"].append("Isolation Forest")
    results["Anomaly Type"].append(anomaly_type)
    results["Accuracy"].append(accuracy_iso)
    results["Precision"].append(precision_iso)
    results["Recall"].append(recall_iso)
    results["F1 Score"].append(f1_iso)

    # LOF Model Evaluation
    lof.fit(simulated_data[features])
    lof_preds = lof.predict(simulated_data[features])
    lof_preds = (lof_preds == -1).astype(int)
    
    accuracy_lof = accuracy_score(simulated_data['anomaly'], lof_preds)
    precision_lof = precision_score(simulated_data['anomaly'], lof_preds)
    recall_lof = recall_score(simulated_data['anomaly'], lof_preds)
    f1_lof = f1_score(simulated_data['anomaly'], lof_preds)
    
    # Log LOF results
    results["Model"].append("LOF")
    results["Anomaly Type"].append(anomaly_type)
    results["Accuracy"].append(accuracy_lof)
    results["Precision"].append(precision_lof)
    results["Recall"].append(recall_lof)
    results["F1 Score"].append(f1_lof)

# Convert results to DataFrame for readability
results_df = pd.DataFrame(results)
print("Model Performance with Different Types of Simulated Anomalies:\n", results_df)
