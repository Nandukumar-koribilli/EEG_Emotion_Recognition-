import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

import matplotlib.pyplot as plt


# Load data
data = np.load("seed_dataset/DatasetCaricatoNoImage.npz")

labels = np.load("seed_dataset/LabelsNoImage.npz")

subjects = np.load("seed_dataset/SubjectsNoImage.npz")


# Access arrays
X = data['arr_0']

y = labels['arr_0']

s = subjects['arr_0']


# Print dataset info
print("EEG Data Shape:", X.shape)

print("Labels Shape:", y.shape)

print("Subjects Shape:", s.shape)


# Flatten features
X = X.reshape(X.shape[0], -1)

print("\nFlattened Shape:", X.shape)


# Feature Selection
selector = SelectKBest(score_func=f_classif, k=50)

X_selected = selector.fit_transform(X, y)

print("Selected Features Shape:", X_selected.shape)


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_selected,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Train Model
model = RandomForestClassifier(n_estimators=100)

model.fit(X_train, y_train)


# Predictions
predictions = model.predict(X_test)


# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", accuracy)


# Classification Report
print("\nClassification Report:\n")

print(classification_report(y_test, predictions))


# Confusion Matrix
print("\nConfusion Matrix:\n")

print(confusion_matrix(y_test, predictions))


# Feature Importance Graph
importance = model.feature_importances_

plt.figure(figsize=(12,6))

plt.bar(range(len(importance)), importance)

plt.title("SEED Feature Importance")

plt.xlabel("Feature Index")

plt.ylabel("Importance")

plt.tight_layout()

plt.show()