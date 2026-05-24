import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from train_model import train_random_forest
from train_model import train_svm
from train_model import train_xgboost

from evaluate import evaluate_model


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

plt.suptitle("SEED Dataset - Feature Importance Comparison", fontsize=14, fontweight='bold')

plt.tight_layout()

plt.show()