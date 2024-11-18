from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Hyperparameter tuning
param_grid_lr = {
    'C': [0.1, 1.0, 10],  # Regularization strength
    'solver': ['liblinear', 'lbfgs']  # Solvers for convergence
}

lr = LogisticRegression(random_state=42, max_iter=1000)
grid_lr = GridSearchCV(lr, param_grid_lr, cv=2, scoring='accuracy')
grid_lr.fit(X_original_scaled, y_original)

# Best estimator evaluation
best_lr = grid_lr.best_estimator_
print("Best LR Parameters:", grid_lr.best_params_)

# Evaluate on original dataset
y_pred_original = best_lr.predict(X_original_scaled)
print("\nLogistic Regression on Original Dataset")
print("Accuracy:", accuracy_score(y_original, y_pred_original))
print("Precision:", precision_score(y_original, y_pred_original, zero_division=0))
print("Recall:", recall_score(y_original, y_pred_original))
print("F1 Score:", f1_score(y_original, y_pred_original))

# Evaluate on augmented dataset
y_pred_augmented = best_lr.predict(X_augmented_scaled)
print("\nLogistic Regression on Augmented Dataset")
print("Accuracy:", accuracy_score(y_augmented, y_pred_augmented))
print("Precision:", precision_score(y_augmented, y_pred_augmented, zero_division=0))
print("Recall:", recall_score(y_augmented, y_pred_augmented))
print("F1 Score:", f1_score(y_augmented, y_pred_augmented))
