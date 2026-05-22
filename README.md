# EEG Emotion Recognition using DEAP and SEED Datasets

This project implements an EEG-based emotion recognition system using machine learning and data mining techniques. It preprocesses EEG signals, extracts multi-view features, applies hybrid feature selection, and trains classifiers to recognize emotional states.

The project supports two datasets:

- DEAP Dataset for binary valence classification
- SEED Dataset for multi-class emotion classification

## Project Overview

The system performs:

- EEG preprocessing
- Multi-view feature extraction
- Hybrid feature selection
- Emotion classification
- Feature importance analysis

## Datasets Used

### 1. DEAP Dataset

Emotion type:

- Binary classification

Predicted labels:

- High valence
- Low valence

Dataset format:

- `.dat`

### 2. SEED Dataset

Emotion type:

- Multi-class classification

Predicted labels:

- Positive emotion
- Neutral emotion
- Negative emotion

Dataset format:

- `.npz`

## Project Structure

```text
EEG-EMOTION-PROJECT/
│
├── deap_dataset/
│   ├── s01.dat
│   ├── s02.dat
│   └── ...
│
├── seed_dataset/
│   ├── DatasetCaricatoNoImage.npz
│   ├── LabelsNoImage.npz
│   └── SubjectsNoImage.npz
│
├── preprocessing.py
├── feature_extraction.py
├── feature_selection.py
├── train_model.py
├── evaluate.py
│
├── main_deap.py
├── main_seed.py
│
└── README.md
```

## Features Implemented

### EEG Preprocessing

- Bandpass filtering
- Signal normalization

### Multi-View Feature Extraction

#### Time-Domain Features

- Mean
- Variance
- Standard deviation
- RMS
- Skewness
- Kurtosis

#### Frequency-Domain Features

- Alpha power
- Beta power
- Theta power
- Gamma power

#### Differential Entropy

Formula:

$$
DE = \frac{1}{2} \ln(2\pi e\sigma^2)
$$

### Hybrid Feature Selection

- SelectKBest
- ANOVA feature selection

### Machine Learning Models

- Random Forest
- SVM
- XGBoost

## Installation

Install the required libraries:

```bash
pip install numpy pandas scipy matplotlib seaborn scikit-learn mne xgboost shap antropy
```

## Running DEAP Dataset

Run:

```bash
python main_deap.py
```

### DEAP Outputs

Generated outputs:

- Accuracy
- Macro-F1 Score
- Balanced Accuracy
- ROC-AUC
- Confusion Matrix
- Feature Importance Graph

### DEAP Results

- Dataset shape: `(1280, 352)`
- Selected features: `(1280, 50)`
- Accuracy: `0.77734375`
- Macro-F1 Score: `0.50`
- Balanced Accuracy: `0.52`
- ROC-AUC: `0.5197103043637696`

## Running SEED Dataset

Run:

```bash
python main_seed.py
```

### SEED Outputs

Generated outputs:

- Accuracy
- Macro-F1 Score
- Balanced Accuracy
- Confusion Matrix
- Feature Importance Graph

### SEED Results

- EEG data shape: `(50910, 5, 62)`
- Flattened feature shape: `(50910, 310)`
- Selected features: `(50910, 50)`
- Accuracy: `1.0`
- Macro-F1 Score: `1.00`
- Balanced Accuracy: `1.00`

### Classification Report

| Class | Precision | Recall | F1-Score |
| --- | --- | --- | --- |
| 0 | 1.00 | 1.00 | 1.00 |
| 1 | 1.00 | 1.00 | 1.00 |
| 2 | 1.00 | 1.00 | 1.00 |

### Confusion Matrix

```text
[[3360    0    0]
 [   0 3312    0]
 [   0    0 3510]]
```

## Workflow

```text
EEG Dataset
     ↓
Preprocessing
     ↓
Feature Extraction
     ↓
Feature Selection
     ↓
Machine Learning Classification
     ↓
Emotion Prediction
```

## Evaluation Metrics

The project computes:

- Accuracy
- Macro-F1 Score
- Balanced Accuracy
- ROC-AUC
- Confusion Matrix

## Feature Importance

Feature importance graphs are generated using the Random Forest classifier to:

- Identify important EEG features
- Improve explainability

## Future Improvements

- Deep learning models
- CNN-based EEG classification
- Transformer models
- SHAP explainability
- Real-time EEG emotion recognition

## Notes

- The DEAP pipeline focuses on binary valence classification.
- The SEED pipeline supports multi-class emotion classification.
- Make sure the dataset files are present in `deap_dataset/` and `seed_dataset/` before running the scripts.