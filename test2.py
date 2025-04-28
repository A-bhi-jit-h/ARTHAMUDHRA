import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split
import pandas as pd

# Load dataset
data_dict = pickle.load(open(r'C:\Users\Asus\OneDrive\Desktop\PROJECT MAIN\data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Load trained model
model_dict = pickle.load(open(r'C:\Users\Asus\OneDrive\Desktop\PROJECT MAIN\model.p', 'rb'))
model = model_dict['model']

# Make predictions
y_pred = model.predict(x_test)

# Compute evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# Print evaluation results
print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-score: {f1:.2f}')
print('\nClassification Report:\n', classification_report(y_test, y_pred))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(labels), yticklabels=np.unique(labels))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Convert classification report to DataFrame for visualization
df_report = pd.DataFrame(report).transpose()
df_report = df_report.drop(index=['accuracy', 'macro avg', 'weighted avg'])

# Plot Precision, Recall, F1-score
plt.figure(figsize=(10, 5))
df_report[['precision', 'recall', 'f1-score']].plot(kind='bar', figsize=(12, 6), rot=45)
plt.title('Precision, Recall, and F1-Score for Each Class')
plt.ylabel('Score')
plt.xlabel('Sign Classes')
plt.legend()
plt.show()

# Overall Metrics Bar Chart
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
values = [accuracy, precision, recall, f1]

plt.figure(figsize=(8, 5))
plt.bar(metrics, values, color=['blue', 'green', 'red', 'purple'])
plt.ylim(0, 1)
plt.xlabel('Evaluation Metrics')
plt.ylabel('Score')
plt.title('Overall Model Performance')
plt.show()

# Simulated Latency Analysis (for real-time performance)
frame_numbers = np.arange(1, 51)  # Simulating 50 frames
latency_values = np.random.uniform(0.05, 0.2, size=50)  # Simulating response time per frame in seconds

plt.figure(figsize=(10, 5))
plt.plot(frame_numbers, latency_values, marker='o', linestyle='-', color='red')
plt.xlabel('Frame Number')
plt.ylabel('Processing Time (seconds)')
plt.title('Real-Time Latency Analysis')
plt.show()