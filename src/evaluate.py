import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def evaluate_classification_metrics(
    name: str,
    model: object,
    X_test,
    y_test
) -> dict:

    """
    Evaluate classification model performance.

    Parameters:
    ----------
    name : str
        Model name

    model : trained ML model
        Trained classifier

    X_test : array-like
        Test features

    y_test : array-like
        Actual labels

    Returns:
    -------
    dict
        Updated results list
    """

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    roc_auc = roc_auc_score(
        y_test,
        model.predict_proba(X_test)[:, 1]
    )

    print("=" * 50)
    print(name)
    print("=" * 50)

    print(f"Accuracy  : {accuracy}")
    print(f"Precision : {precision}")
    print(f"Recall    : {recall}")
    print(f"F1 Score  : {f1}")
    print(f"ROC-AUC   : {roc_auc}")

    print("\nClassification Report\n")

    print(classification_report(y_test, predictions))

    cm = confusion_matrix(y_test, predictions)

    ConfusionMatrixDisplay(confusion_matrix=cm).plot()

    plt.title(f"Confusion Matrix - {name}")
    plt.show()
        
    return {
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC-AUC': roc_auc
    }


def evaluate_regression_model(
    name: str,
    model: object,
    X_test,
    y_test
):
    """
    Evaluate Regression model performance.

    Parameters:
    ----------
    name : str
        Model name

    model : trained ML model
        Trained classifier

    X_test : array-like
        Test features

    y_test : array-like
        Actual labels

    Returns:
    -------
    dict
        Updated results list
    """

    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)

    mse = mean_squared_error(y_test, predictions)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, predictions)

    print("=" * 50)
    print(name)
    print("=" * 50)

    print(f"MAE  : {mae}")
    print(f"MSE  : {mse}")
    print(f"RMSE : {rmse}")
    print(f"R2   : {r2}")

    return {
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }