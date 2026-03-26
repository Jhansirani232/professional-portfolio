import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle
import shap  # For model explainability

# Load dataset
df = pd.read_csv('our.csv')

# Label encode 'Type' column
label_encoder = LabelEncoder()
df['Type'] = label_encoder.fit_transform(df['Type'])

# Drop unnecessary columns
df = df.drop(columns=['Product ID'])

# Define features and target
X = df.drop(["Machine failure", "UDI"], axis=1)  # X should have 11 features
y = df["Machine failure"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Standardize the feature data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X_train_scaled, y_train)

# Save the trained model and scaler to .pkl files
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

# SHAP for explainability
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_scaled)

# Save the SHAP explainer for later use
with open("shap_explainer.pkl", "wb") as file:
    pickle.dump(explainer, file)

# Test the model
accuracy = model.score(X_test_scaled, y_test)
print(f"Model accuracy: {accuracy}")