from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Model initialization
nb = GaussianNB()

# Fit model on original data
nb.fit(X_original_scaled, y_original)

# Evaluate on original dataset
y_pred_original = nb.predict(X_original_scaled)
print("\nNaive Bayes on Original Dataset")
print("Accuracy:", accuracy_score(y_original, y_pred_original))
print("Precision:", precision_score(y_original, y_pred_original, zero_division=0))
print("Recall:", recall_score(y_original, y_pred_original))
print("F1 Score:", f1_score(y_original, y_pred_original))

# Evaluate on augmented dataset
y_pred_augmented = nb.predict(X_augmented_scaled)
print("\nNaive Bayes on Augmented Dataset")
print("Accuracy:", accuracy_score(y_augmented, y_pred_augmented))
print("Precision:", precision_score(y_augmented, y_pred_augmented, zero_division=0))
print("Recall:", recall_score(y_augmented, y_pred_augmented))
print("F1 Score:", f1_score(y_augmented, y_pred_augmented))
