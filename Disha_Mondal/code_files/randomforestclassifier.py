from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# Preprocessing
X_original = df.drop(columns=['label'])
y_original = df['label']
X_augmented = augmented_data.drop(columns=['label'])
y_augmented = augmented_data['label']

# Ensure consistent columns for scaling
common_columns = X_original.columns.intersection(X_augmented.columns)
X_original = X_original[common_columns]
X_augmented = X_augmented[common_columns]

# Scaling
scaler = StandardScaler()
X_original_scaled = scaler.fit_transform(X_original)
X_augmented_scaled = scaler.transform(X_augmented)

# Simplified Hyperparameter Tuning
param_grid_rf = {
    'n_estimators': [50, 100],  # fewer values for quicker initial run
    'max_depth': [10, None],    # fewer values for quicker initial run
    'max_features': ['sqrt']    # single value to reduce combinations
}

rf = RandomForestClassifier(random_state=42)
grid_rf = GridSearchCV(rf, param_grid_rf, cv=2, scoring='accuracy')  # Reduced cv folds to 2 for faster results
grid_rf.fit(X_original_scaled, y_original)

# Best estimator evaluation
best_rf = grid_rf.best_estimator_
print("Best RF Parameters:", grid_rf.best_params_)

# Evaluate on original dataset
y_pred_original = best_rf.predict(X_original_scaled)
print("Random Forest on Original Dataset")
print("Accuracy:", accuracy_score(y_original, y_pred_original))
print("Precision:", precision_score(y_original, y_pred_original, zero_division=0))
print("Recall:", recall_score(y_original, y_pred_original))
print("F1 Score:", f1_score(y_original, y_pred_original))

# Evaluate on augmented dataset
y_pred_augmented = best_rf.predict(X_augmented_scaled)
print("Random Forest on Augmented Dataset")
print("Accuracy:", accuracy_score(y_augmented, y_pred_augmented))
print("Precision:", precision_score(y_augmented, y_pred_augmented, zero_division=0))
print("Recall:", recall_score(y_augmented, y_pred_augmented))
print("F1 Score:", f1_score(y_augmented, y_pred_augmented))
