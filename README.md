# EEG Emotion Recognition using Multi-View Feature Extraction and Hybrid Feature Selection

This project implements an EEG-based emotion recognition pipeline using the DEAP dataset. It preprocesses EEG signals, extracts multi-view features, performs hybrid feature selection, and trains machine learning models to classify emotional states.

The current implementation predicts **binary valence** labels from EEG brain signals:

- `1` if valence > 5
- `0` otherwise

## Features

- EEG signal preprocessing
- Time-domain feature extraction
- Frequency-domain feature extraction
- Differential entropy calculation
- Hybrid feature selection
- Emotion classification
- Accuracy evaluation
- Confusion matrix generation
- ROC-AUC evaluation
- Feature importance visualization

## Dataset

This project uses the **DEAP dataset**.

Download the dataset from:

https://www.eecs.qmul.ac.uk/mmv/datasets/deap/

Place the downloaded `.dat` files inside the `dataset/` folder:

```text
dataset/
├── s01.dat
├── s02.dat
├── s03.dat
└── ...
```

## Project Structure

```text
EEG_Emotion_Project/
├── dataset/
├── preprocessing.py
├── feature_extraction.py
├── feature_selection.py
├── train_model.py
├── evaluate.py
├── main.py
└── README.md
```

## Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd EEG_Emotion_Project
```

### Step 2: Install Python Libraries

```bash
pip install numpy pandas scipy matplotlib seaborn scikit-learn mne xgboost shap antropy
```

## Running the Project

Run the full pipeline with:

```bash
python main.py
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

## Implemented Pipeline

The current `main.py` script performs the following steps:

1. Loads all `.dat` files from the `dataset/` folder.
2. Extracts EEG data and valence labels.
3. Converts valence into a binary emotion label.
4. Applies preprocessing to each EEG channel.
5. Extracts features from the first 32 EEG channels.
6. Concatenates all channel features into one feature vector.
7. Selects the top 50 features.
8. Splits the data into training and test sets.
9. Trains a Random Forest model.
10. Evaluates the model and plots feature importance.

## Feature Extraction

### Time-Domain Features

- Mean
- Variance
- Standard Deviation
- RMS
- Skewness
- Kurtosis

### Frequency-Domain Features

- Alpha power
- Beta power
- Theta power
- Gamma power

### Differential Entropy

The differential entropy is calculated as:

$$
DE = \frac{1}{2} \ln(2\pi e\sigma^2)
$$

## Machine Learning Models

The project includes support for the following classifiers:

- Random Forest
- SVM
- XGBoost

## Evaluation Metrics

The project reports the following metrics:

- Accuracy
- Macro-F1 Score
- Balanced Accuracy
- ROC-AUC
- Confusion Matrix

## Obtained Results

Reported results for the current pipeline:

- Dataset shape: `(1280, 352)`
- Selected features: `(1280, 50)`
- Accuracy: `77.7%`
- ROC-AUC: `0.519`

## Output Graphs

The project generates:

- Feature importance graph
- Confusion matrix
- Classification report

## Future Improvements

- Deep learning models
- CNN-based EEG classification
- Transformer-based EEG analysis
- SHAP explainability analysis
- Real-time EEG emotion recognition

## Notes

- The current entrypoint focuses on binary valence classification.
- The `train_model.py` module includes Random Forest, SVM, and XGBoost training functions.
- Make sure the `.dat` files are present in `dataset/` before running the project.