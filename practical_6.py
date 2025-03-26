import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay

# Load the dataset
df = pd.read_csv("diabetes.csv")

# Selecting features and target variable
X = df.iloc[:, :-1]  # Feature columns
y = df.iloc[:, -1]   # Target column

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the Naïve Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Compute Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
print("Confusion Matrix:")
print(cm)

# Extract TP, FP, TN, FN
TP = np.diag(cm)  # True Positives (diagonal values)
FP = np.sum(cm, axis=0) - TP  # False Positives (column sum minus TP)
FN = np.sum(cm, axis=1) - TP  # False Negatives (row sum minus TP)
TN = np.sum(cm) - (TP + FP + FN)  # True Negatives (remaining values)

# Compute metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')  # Macro-averaged precision
recall = recall_score(y_test, y_pred, average='macro')  # Macro-averaged recall
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

# Plot the Confusion Matrix
display_labels = model.classes_
conf_matrix = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
conf_matrix.plot(cmap=plt.cm.Blues)
plt.show()

# Show probability estimates for some rows
probabilities = model.predict_proba(X_test[:5])  # Show probabilities for the first 5 rows
print("\nProbability Estimates for First 5 Rows:")
print(probabilities)
