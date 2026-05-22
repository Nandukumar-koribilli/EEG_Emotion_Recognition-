from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Train Random Forest
def train_random_forest(X_train, y_train):

    model = RandomForestClassifier(n_estimators=100)

    model.fit(X_train, y_train)

    return model


# Train SVM
def train_svm(X_train, y_train):

    model = SVC(probability=True)

    model.fit(X_train, y_train)

    return model


# Train XGBoost
def train_xgboost(X_train, y_train):

    model = XGBClassifier()

    model.fit(X_train, y_train)

    return model