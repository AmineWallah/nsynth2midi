from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    recall_score,
    precision_score
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib
import numpy as np

def main():
    X = np.load('X.npy')
    y = np.load('y.npy')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')

    X_test = scaler.transform(X_test)
    y_pred = clf.predict(X_test)

    # Overall metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)

    print(f"Accuracy:           {accuracy:.4f}")
    print(f"F1 (macro avg):     {f1_macro:.4f}")
    print(f"F1 (weighted avg):  {f1_weighted:.4f}")
    print(f"Recall (macro avg): {recall_macro:.4f}")
    print(f"Precision (macro avg): {precision_macro:.4f}")
    print()
    print(classification_report(y_test, y_pred, zero_division=0))

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(30, 30))
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    plt.xticks(rotation=90)
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.show()

if __name__ == '__main__':
    main()