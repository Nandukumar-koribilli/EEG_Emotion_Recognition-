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
from train_model import train_svm
from train_model import train_xgboost

from evaluate import evaluate_model


# Store all data
all_features = []
all_labels = []

dataset_path = "deap_dataset"


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


# ============================================================
# 1. Random Forest
# ============================================================
print("\n" + "=" * 60)
print("  RANDOM FOREST")
print("=" * 60)

print("\nTraining Random Forest Model...")

rf_model = train_random_forest(X_train, y_train)

print("\nEvaluating Random Forest...\n")

rf_acc = evaluate_model(rf_model, X_test, y_test)


# ============================================================
# 2. SVM
# ============================================================
print("\n" + "=" * 60)
print("  SVM (Support Vector Machine)")
print("=" * 60)

print("\nTraining SVM Model...")

svm_model = train_svm(X_train, y_train)

print("\nEvaluating SVM...\n")

svm_acc = evaluate_model(svm_model, X_test, y_test)


# ============================================================
# 3. XGBoost
# ============================================================
print("\n" + "=" * 60)
print("  XGBOOST")
print("=" * 60)

print("\nTraining XGBoost Model...")

xgb_model = train_xgboost(X_train, y_train)

print("\nEvaluating XGBoost...\n")

xgb_acc = evaluate_model(xgb_model, X_test, y_test)


# ============================================================
# Feature Importance Graphs
# ============================================================
print("\nGenerating Feature Importance Graphs...")

fig, axes = plt.subplots(1, 3, figsize=(24, 6))

# Random Forest Feature Importance
axes[0].bar(range(len(rf_model.feature_importances_)), rf_model.feature_importances_)
axes[0].set_title("Random Forest - Feature Importance")
axes[0].set_xlabel("Feature Index")
axes[0].set_ylabel("Importance")

# Accuracy Comparison (all 3 models)
model_names = ['Random Forest', 'SVM', 'XGBoost']
accuracies = [rf_acc, svm_acc, xgb_acc]
colors = ['#2196F3', '#FF9800', '#4CAF50']
bars = axes[1].bar(model_names, accuracies, color=colors)
axes[1].set_title("Model Accuracy Comparison")
axes[1].set_ylabel("Accuracy")
axes[1].set_ylim(0, 1.1)
for bar, acc in zip(bars, accuracies):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{acc:.4f}', ha='center', fontweight='bold')

# XGBoost Feature Importance
axes[2].bar(range(len(xgb_model.feature_importances_)), xgb_model.feature_importances_)
axes[2].set_title("XGBoost - Feature Importance")
axes[2].set_xlabel("Feature Index")
axes[2].set_ylabel("Importance")

plt.suptitle("DEAP Dataset - Feature Importance Comparison", fontsize=14, fontweight='bold')

plt.tight_layout()

plt.show()