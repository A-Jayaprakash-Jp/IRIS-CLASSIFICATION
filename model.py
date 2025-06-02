# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset from external CSV file
# Replace 'iris.csv' with your actual path or URL
df = pd.read_csv('Task_1/IRIS.csv')

# Display the first few rows to confirm it's loaded correctly
print("Dataset Preview:\n", df.head())

# Separate features and target
X = df.drop("species", axis=1)  # 'species' is the target column
y = df["species"]

# Encode categorical target labels to numeric
le = LabelEncoder()
y = le.fit_transform(y)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Random Forest Classifier
model = RandomForestClassifier()
model.fit(X_train_scaled, y_train)

# Predict on test data
y_pred = model.predict(X_test_scaled)

# Evaluate the model
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
import joblib
# Export/save the trained model
joblib.dump(model, 'iris_model.pkl')

# Optional: Export the label encoder and scaler too
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')