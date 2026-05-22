import pickle
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from preprocessing import bandpass_filter
from preprocessing import normalize_data

from feature_extraction import extract_features

from feature_selection import select_features

from train_model import train_random_forest

from evaluate import evaluate_model


# Store all data
all_features = []
all_labels = []

dataset_path = "dataset"


# Load all .dat files
for filename in os.listdir(dataset_path):

    if filename.endswith(".dat"):

        print(f"Loading {filename}...")

        file_path = os.path.join(dataset_path, filename)

        file = open(file_path, "rb")

        data = pickle.load(file, encoding='latin1')

        eeg_data = data['data']
        labels = data['labels']

        # Valence labels
        valence = labels[:, 0]

        binary_labels = []

        for v in valence:

            if v > 5:
                binary_labels.append(1)
            else:
                binary_labels.append(0)

        # Feature extraction
        for i, trial in enumerate(eeg_data):

            channel_features = []

            # Use first 32 EEG channels
            for channel in trial[:32]:

                # Preprocessing
                filtered = bandpass_filter(channel)

                normalized = normalize_data(filtered)

                # Feature Extraction
                features = extract_features(normalized)

                channel_features.extend(features)

            all_features.append(channel_features)

            all_labels.append(binary_labels[i])


# Convert to numpy arrays
X = np.array(all_features)

y = np.array(all_labels)

print("\nDataset Shape:", X.shape)
print("Labels Shape:", y.shape)


# Feature Selection
print("\nPerforming Feature Selection...")

X_selected, selector = select_features(X, y, k=50)

print("Selected Feature Shape:", X_selected.shape)


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_selected,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Train Model
print("\nTraining Random Forest Model...")

model = train_random_forest(X_train, y_train)


# Evaluate Model
print("\nEvaluating Model...\n")

evaluate_model(model, X_test, y_test)


# Feature Importance Graph
print("\nGenerating Feature Importance Graph...")

importance = model.feature_importances_

plt.figure(figsize=(12, 6))

plt.bar(range(len(importance)), importance)

plt.title("Feature Importance")

plt.xlabel("Feature Index")

plt.ylabel("Importance")

plt.tight_layout()

plt.show()