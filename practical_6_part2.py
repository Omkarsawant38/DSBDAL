import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("bank_transactions_data_2.csv")


# Drop unnecessary columns
columns_to_drop = ['TransactionID', 'AccountID', 'DeviceID', 'IP Address']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Encode categorical variables
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store encoder for reference

# Define features (X) and target (y)
X = df.drop(columns=['TransactionType'])  # Features
y = df['TransactionType']  # Target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Naïve Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Compute Confusion Matrix
labels = sorted(np.unique(y_test))  # Ensure correct labels
cm = confusion_matrix(y_test, y_pred, labels=labels)

# Extract TP, FP, TN, FN
TP = np.diag(cm)
FP = np.sum(cm, axis=0) - TP
FN = np.sum(cm, axis=1) - TP
TN = np.sum(cm) - (TP + FP + FN)

# Compute evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
error_rate = 1 - accuracy

# Display results
print(f"Accuracy: {accuracy:.4f}")
print(f"Error Rate: {error_rate:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"TP: {TP}")
print(f"FP: {FP}")
print(f"TN: {TN}")
print(f"FN: {FN}")

# Plot Confusion Matrix
conf_matrix = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
fig, ax = plt.subplots(figsize=(8, 6))
conf_matrix.plot(cmap=plt.cm.Blues, ax=ax)
plt.xticks(rotation=90)
plt.show()

# Show probability estimates for some rows
probabilities = model.predict_proba(X_test[:5])
print("\nProbability Estimates for First 5 Rows:")
print(probabilities)