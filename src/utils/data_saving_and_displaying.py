import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def display_npy_file(file_path):
    # Load the .npy file
    data = np.load(file_path)

    # Convert to a DataFrame for better readability
    df = pd.DataFrame(data)

    # Print the DataFrame
    print(df)

    # Optionally, save to a CSV file for viewing in an editor
    csv_file_path = file_path.replace('.npy', '.csv')
    df.to_csv(csv_file_path, index=False)
    print(f"Data saved to {csv_file_path}")
    # print mean and std
    print(f"Mean: {np.mean(data)}")
    print(f"Std: {np.std(data)}")


def save_and_display_results(test_actuals, test_predictions, subfolder):
    """Save the actual and predicted values, and convert them to CSV figures."""
    actuals_path = os.path.join(subfolder, 'test_actuals.npy')
    predictions_path = os.path.join(subfolder, 'test_predictions.npy')

    # Save as .npy figures
    np.save(actuals_path, np.array(test_actuals))
    np.save(predictions_path, np.array(test_predictions))

    # Convert .npy to .csv
    display_npy_file(actuals_path)
    display_npy_file(predictions_path)


def save_and_display_results_classification(test_actuals, test_predictions, subfolder, dataset='test'):
    """Save and display the results of the classification model."""

    # Calculate metrics
    accuracy = accuracy_score(test_actuals, test_predictions)
    precision = precision_score(test_actuals, test_predictions, average='weighted')
    recall = recall_score(test_actuals, test_predictions, average='weighted')
    f1 = f1_score(test_actuals, test_predictions, average='weighted')

    # Create a DataFrame with actual and predicted values
    results_df = pd.DataFrame({
        'Actual': test_actuals,
        'Predicted': test_predictions
    })

    # Save results to CSV
    results_path = os.path.join(subfolder, f'classification_results_{dataset}.csv')
    results_df.to_csv(results_path, index=False)

    # Save metrics
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Value': [accuracy, precision, recall, f1]
    })
    metrics_path = os.path.join(subfolder, f'classification_metrics_{dataset}.csv')
    metrics_df.to_csv(metrics_path, index=False)

    # Create and save confusion matrix
    cm = confusion_matrix(test_actuals, test_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    cm_path = os.path.join(subfolder, f'confusion_matrix_{dataset}.png')
    plt.savefig(cm_path)
    plt.close()

    # Print results
    print(f"Results saved to: {results_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Confusion matrix saved to: {cm_path}")
    print("\nClassification Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")