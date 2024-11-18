import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data for Model Performance
model_performance_data = {
    "Model": [
        "Random Forest", "Logistic Regression", "K-Nearest Neighbors", 
        "Support Vector Machine", "Gradient Boosting", "XGBoost", 
        "Naive Bayes", "Decision Tree","Random Forest", "Logistic Regression", "K-Nearest Neighbors", 
        "Support Vector Machine", "Gradient Boosting", "XGBoost", 
        "Naive Bayes", "Decision Tree"
    ],
    "Dataset": [
        "Original", "Original", "Original", "Original", "Original", "Original", "Original", "Original",
        "Augmented", "Augmented", "Augmented", "Augmented", "Augmented", "Augmented", "Augmented", "Augmented"
    ],
    "Accuracy": [
        0.9844, 0.8641, 1.0, 0.8756, 0.9799, 0.9176, 0.6392, 0.9850,
        0.4043, 0.4246, 0.4094, 0.4325, 0.4021, 0.3492, 0.7294, 0.4045
    ],
    "Precision": [
        0.9828, 0.8957, 1.0, 0.9096, 0.9774, 0.8904, 0.9253, 0.9831,
        0.0077, 0.0078, 0.0081, 0.0121, 0.0077, 0.0077, 0.0251, 0.0086
    ],
    "Recall": [
        0.9910, 0.8722, 1.0, 0.8773, 0.9891, 0.9818, 0.4255, 0.9918,
        0.66, 0.65, 0.69, 1.0, 0.66, 0.72, 1.0, 0.74
    ],
    "F1 Score": [
        0.9869, 0.8838, 1.0, 0.8932, 0.9832, 0.9339, 0.5829, 0.9874,
        0.0152, 0.0155, 0.0160, 0.0240, 0.0152, 0.0152, 0.0490, 0.0170
    ]
}

# Convert to DataFrame
df_performance = pd.DataFrame(model_performance_data)
