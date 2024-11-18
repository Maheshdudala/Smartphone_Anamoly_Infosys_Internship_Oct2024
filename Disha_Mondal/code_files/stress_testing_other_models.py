from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define models with best parameters
models = {
    "Random Forest": RandomForestClassifier(max_depth=10, max_features='sqrt', n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(C=0.1, solver='liblinear', random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=10, p=1, weights='distance'),
    "Support Vector Machine": SVC(C=0.1, kernel='rbf', random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(learning_rate=0.05, max_depth=5, n_estimators=50, random_state=42),
    "XGBoost": XGBClassifier(learning_rate=0.01, max_depth=3, n_estimators=50, random_state=42),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_split=10, random_state=42)
}

# Results storage
stress_test_results = []

# Scale stress-tested data
X_stress_test_scaled = scaler.transform(df_stress_test[common_columns])

# Evaluate models on stressed data
for model_name, model in models.items():
    for anomaly_type in anomaly_types.keys():
        # Train on original data
        model.fit(X_original_scaled, y_original)
        
        # Predict on stress-tested data
        y_pred_stressed = model.predict(X_stress_test_scaled)

        # Calculate performance metrics
        accuracy = accuracy_score(df_stress_test['label'], y_pred_stressed)
        precision = precision_score(df_stress_test['label'], y_pred_stressed, zero_division=1)
        recall = recall_score(df_stress_test['label'], y_pred_stressed, zero_division=1)
        f1 = f1_score(df_stress_test['label'], y_pred_stressed, zero_division=1)

        # Append results
        stress_test_results.append({
            "Model": model_name,
            "Anomaly Type": anomaly_type,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

# Convert results to DataFrame
df_stress_results = pd.DataFrame(stress_test_results)
print(df_stress_results)
