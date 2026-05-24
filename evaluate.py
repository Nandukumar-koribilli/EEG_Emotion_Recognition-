from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

# Evaluate model
def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("Accuracy:", accuracy)

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    try:
        roc = roc_auc_score(y_test, predictions)
        print("\nROC-AUC:", roc)
    except:
        pass

    return accuracy