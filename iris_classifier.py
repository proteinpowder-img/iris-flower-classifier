# Import libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Step 1: Load the Iris dataset
iris = load_iris()
X = iris.data  # Features (measurements)
y = iris.target  # Target (species)

# Create a DataFrame for better understanding
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = iris.target_names[y]

print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset info:")
print(df.info())

# Step 2: Split data into training and testing sets
# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# Step 3: Create and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("\nModel training complete!")

# Step 4: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 5: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2%}")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Step 6: Make predictions on new data
print("\nMaking predictions on new iris measurements:")
new_flower = [[5.1, 3.5, 1.4, 0.2]]  # Example measurements
prediction = model.predict(new_flower)
print(f"Predicted species: {iris.target_names[prediction[0]]}")
