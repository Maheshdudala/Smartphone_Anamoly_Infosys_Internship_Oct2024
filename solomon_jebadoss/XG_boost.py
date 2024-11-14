from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Hyperparameter tuning
param_grid_xgb = {
    'n_estimators': [50, 100],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5]
}

xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
grid_xgb = GridSearchCV(xgb, param_grid_xgb, cv=2, scoring='accuracy')
grid_xgb.fit(X_original_scaled, y_original)

# Best estimator evaluation
best_xgb = grid_xgb.best_estimator_
print("Best XGB Parameters:", grid_xgb.best_params_)

# Evaluate on original dataset
y_pred_original = best_xgb.predict(X_original_scaled)
print("\nXGBoost on Original Dataset")
print("Accuracy:", accuracy_score(y_original, y_pred_original))
print("Precision:", precision_score(y_original, y_pred_original, zero_division=0))
print("Recall:", recall_score(y_original, y_pred_original))
print("F1 Score:", f1_score(y_original, y_pred_original))

# Evaluate on augmented dataset
y_pred_augmented = best_xgb.predict(X_augmented_scaled)
print("\nXGBoost on Augmented Dataset")
print("Accuracy:", accuracy_score(y_augmented, y_pred_augmented))
print("Precision:", precision_score(y_augmented, y_pred_augmented, zero_division=0))
print("Recall:", recall_score(y_augmented, y_pred_augmented))
print("F1 Score:", f1_score(y_augmented, y_pred_augmented))
