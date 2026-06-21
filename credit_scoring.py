import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("credit_data.csv")

print("Dataset Loaded Successfully")
print(data.head())

# Convert categorical columns to numerical values
label_encoder = LabelEncoder()

for column in data.columns:
    if data[column].dtype == "object":
        data[column] = label_encoder.fit_transform(data[column])

# Define target variable
target_column = "credit_risk"

# Features and target
X = data.drop(target_column, axis=1)
y = data[target_column]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nCredit Scoring Accuracy:", accuracy)

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Example prediction
sample_customer = X.iloc[0:1]
prediction = model.predict(sample_customer)

if prediction[0] == 1:
    print("\nPrediction: Good Credit Risk")
else:
    print("\nPrediction: Bad Credit Risk")
