from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Adjusted parameter grid
param_grid_gb = {
    'n_estimators': [50, 100],  # reduced the range for faster tuning
    'learning_rate': [0.1, 0.05],  # common values for learning rate
    'max_depth': [3, 5]  # limited depth for faster training
}

# Gradient Boosting with early stopping
gb = GradientBoostingClassifier(random_state=42)
grid_gb = GridSearchCV(
    gb, param_grid_gb, cv=2, scoring='accuracy', n_jobs=-1  # adjust cv to reduce computation
)
grid_gb.fit(X_original_scaled, y_original)

# Best estimator evaluation
best_gb = grid_gb.best_estimator_
print("Best Gradient Boosting Parameters:", grid_gb.best_params_)

# Evaluate on original dataset
y_pred_original = best_gb.predict(X_original_scaled)
print("\nGradient Boosting on Original Dataset")
print("Accuracy:", accuracy_score(y_original, y_pred_original))
print("Precision:", precision_score(y_original, y_pred_original, zero_division=0))
print("Recall:", recall_score(y_original, y_pred_original))
print("F1 Score:", f1_score(y_original, y_pred_original))

# Evaluate on augmented dataset
y_pred_augmented = best_gb.predict(X_augmented_scaled)
print("\nGradient Boosting on Augmented Dataset")
print("Accuracy:", accuracy_score(y_augmented, y_pred_augmented))
print("Precision:", precision_score(y_augmented, y_pred_augmented, zero_division=0))
print("Recall:", recall_score(y_augmented, y_pred_augmented))
print("F1 Score:", f1_score(y_augmented, y_pred_augmented))