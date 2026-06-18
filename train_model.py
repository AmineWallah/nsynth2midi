import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


def main():
    X = np.load('X.npy')
    y = np.load('y.npy')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model

    clf = RandomForestClassifier(n_jobs=-1 ,random_state=42)
    clf.fit(X_train_scaled, y_train)

    # Save the model and scaler
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(clf, 'model.pkl')



if __name__ == '__main__':
    main()