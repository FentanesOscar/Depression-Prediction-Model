import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.base import clone

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

def one_hot_encode_column(df, column):
    """
    Returns a new DataFrame with one-hot encoding applied to the specified column.
    Returns:
        pd.DataFrame: A new DataFrame with the specified column replaced by one-hot encoded columns.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    
    one_hot = pd.get_dummies(df[column], prefix=column).astype(int)
    df = pd.concat([df.drop(columns=[column]), one_hot], axis=1)
    return df

def model_evaulate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)

    if isinstance(model, GridSearchCV):
        best_model = model.best_estimator_
        print("Best estimator from GridSearchCV:")
        print(best_model)
    else:
        best_model = model

    y_pred = best_model.predict(X_test)
    print_result(y_test, y_pred)


## printing statistics
def print_result(y_test, y_pred):
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Create HTML table
    html_table = f"""
    <table border="1" style="border-collapse: collapse; text-align: center;">
        <tr>
            <th>Metric</th>
            <th>Value</th>
        </tr>
        <tr><td>Accuracy</td><td>{accuracy:.4f}</td></tr>
        <tr><td>Precision</td><td>{precision:.4f}</td></tr>
        <tr><td>Recall</td><td>{recall:.4f}</td></tr>
        <tr><td>F1 Score</td><td>{f1:.4f}</td></tr>
    </table>
    """
    display(HTML(html_table))
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


def crossval_evaluate(model, X, y, n_splits=5, random_state=42):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    best_recall = 0
    best_confusion_matrix = None
    fold_recalls = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        cloned_model = clone(model)
        cloned_model.fit(X_tr, y_tr)

        y_pred = cloned_model.predict(X_val)
        recall = recall_score(y_val, y_pred)
        cm = confusion_matrix(y_val, y_pred)

        fold_recalls.append(recall)

        print(f"\n Fold {fold} Recall: {recall:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        
        if recall > best_recall:
            best_recall = recall
            best_confusion_matrix = cm

        # Display metrics using your existing result function
        print_result(y_val, y_pred)

    print("\n Cross-Validation Summary")
    print(f"All Fold Recalls: {fold_recalls}")
    print(f"Mean Recall: {np.mean(fold_recalls):.4f}")
    print(f"Best Fold Recall: {best_recall:.4f}")
    print(f"Best Confusion Matrix:\n{best_confusion_matrix}")

