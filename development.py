import pandas as pd
import os
import matplotlib.pyplot as plt

# Function to split datasets into training and testing sets
def split_dataset(df, train_ratio=0.8):
    # Shuffle the dataset
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Calculate the split index
    split_index = int(len(df_shuffled) * train_ratio)
    
    # Split the dataset
    train_df = df_shuffled[:split_index]
    test_df = df_shuffled[split_index:]
    
    return train_df, test_df

# Create output directory if it doesn't exist
output_folder = "output_2"
os.makedirs(output_folder, exist_ok=True)

# Load each CSV file
file_paths = {
    "image0": "output/image0.csv",
    "image0_sliding": "output/image0_sliding.csv",
    "image01": "output/image01.csv",
    "image01_sliding": "output/image01_sliding.csv",
    "image1": "output/image1.csv"
}

for name, path in file_paths.items():
    df = pd.read_csv(path)
    
    # Split each dataset into training and testing sets
    train_df, test_df = split_dataset(df)
    
    # Save the results in the output folder
    train_df.to_csv(f"{output_folder}/{name}_train.csv", index=False)
    test_df.to_csv(f"{output_folder}/{name}_test.csv", index=False)

# Load split datasets
# Replace these paths with the actual paths to your training and testing CSV files
train_df_image0 = pd.read_csv("output_2/image0_train.csv")
test_df_image0 = pd.read_csv("output_2/image0_test.csv")
train_df_image0_sliding = pd.read_csv("output_2/image0_sliding_train.csv")
test_df_image0_sliding = pd.read_csv("output_2/image0_sliding_test.csv")
train_df_image01 = pd.read_csv("output_2/image01_train.csv")
test_df_image01 = pd.read_csv("output_2/image01_test.csv")
train_df_image01_sliding = pd.read_csv("output_2/image01_sliding_train.csv")
test_df_image01_sliding = pd.read_csv("output_2/image01_sliding_test.csv")

# Define plotting functions
def plot_histograms(train_df, test_df, feature1, feature2, category_name):
    plt.figure(figsize=(12, 5))
    
    # Histogram for Feature 1
    plt.subplot(1, 2, 1)
    plt.hist(train_df[feature1], bins=30, alpha=0.5, label='Train', color='blue')
    plt.hist(test_df[feature1], bins=30, alpha=0.5, label='Test', color='orange')
    plt.title(f'{category_name} - {feature1} Histogram')
    plt.xlabel(feature1)
    plt.ylabel('Frequency')
    plt.legend()

    # Histogram for Feature 2
    plt.subplot(1, 2, 2)
    plt.hist(train_df[feature2], bins=30, alpha=0.5, label='Train', color='blue')
    plt.hist(test_df[feature2], bins=30, alpha=0.5, label='Test', color='orange')
    plt.title(f'{category_name} - {feature2} Histogram')
    plt.xlabel(feature2)
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_scatter(train_df, test_df, feature1, feature2, category_name):
    plt.figure(figsize=(12, 5))
    
    # Scatter plot for Training Set
    plt.subplot(1, 2, 1)
    plt.scatter(train_df[feature1], train_df[feature2], c=train_df['Label'], cmap='viridis', alpha=0.7)
    plt.title(f'{category_name} - Train Scatter Plot')
    plt.xlabel(feature1)
    plt.ylabel(feature2)

    # Scatter plot for Testing Set
    plt.subplot(1, 2, 2)
    plt.scatter(test_df[feature1], test_df[feature2], c=test_df['Label'], cmap='viridis', alpha=0.7)
    plt.title(f'{category_name} - Test Scatter Plot')
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    
    plt.tight_layout()
    plt.show()
    
# Define a function to handle dense plots by using subsets
def plot_scatter_with_subset(train_df, test_df, feature1, feature2, category_name, max_points=1000):
    # Subset data if it's too dense
    if len(train_df) > max_points:
        train_df = train_df.sample(n=max_points, random_state=42)
    if len(test_df) > max_points:
        test_df = test_df.sample(n=max_points, random_state=42)
    
    plt.figure(figsize=(12, 5))

    # Scatter plot for Training Set
    plt.subplot(1, 2, 1)
    plt.scatter(train_df[feature1], train_df[feature2], c=train_df['Label'], cmap='viridis', alpha=0.7)
    plt.title(f'{category_name} - Train Scatter Plot')
    plt.xlabel(feature1)
    plt.ylabel(feature2)

    # Scatter plot for Testing Set
    plt.subplot(1, 2, 2)
    plt.scatter(test_df[feature1], test_df[feature2], c=test_df['Label'], cmap='viridis', alpha=0.7)
    plt.title(f'{category_name} - Test Scatter Plot')
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    
    plt.tight_layout()
    plt.show()

# Select features and plot
feature_1, feature_2 = '10', '20'

category_name = "image 0 block vectors"
plot_histograms(train_df_image0, test_df_image0, feature_1, feature_2, category_name)
plot_scatter(train_df_image0, test_df_image0, feature_1, feature_2, category_name)

category_name = "image 0 sliding block vectors"
plot_histograms(train_df_image0_sliding, test_df_image0_sliding, feature_1, feature_2, category_name)
plot_scatter(train_df_image0_sliding, test_df_image0_sliding, feature_1, feature_2, category_name)
plot_scatter_with_subset(train_df_image0_sliding, test_df_image0_sliding, feature_1, feature_2, category_name)

category_name = "Merged image 0 and image 1 block vectors"
plot_histograms(train_df_image01, test_df_image01, feature_1, feature_2, category_name)
plot_scatter(train_df_image01, test_df_image01, feature_1, feature_2, category_name)

category_name = "Merged image 0 and image 1 sliding block vector"
plot_histograms(train_df_image01_sliding, test_df_image01_sliding, feature_1, feature_2, category_name)
plot_scatter(train_df_image01_sliding, test_df_image01_sliding, feature_1, feature_2, category_name)
category_name = "Merged image 0 and image 1 sliding block vector"
plot_scatter_with_subset(train_df_image01_sliding, test_df_image01_sliding, feature_1, feature_2, category_name)

from sklearn.linear_model import LassoCV
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import numpy as np
import os
import pandas as pd

# Load the merged training datasets
train_df_image01 = pd.read_csv("output_2/image01_train.csv")
train_df_image01_sliding = pd.read_csv("output_2/image01_sliding_train.csv")

# Function to train lasso regression as a two-class classifier
def train_lasso_classifier(train_df):
    X_train = train_df.drop(columns=['Label'])
    y_train = train_df['Label']

    # LassoCV automatically finds the best alpha value for regularization
    model = LassoCV(cv=5, random_state=42, max_iter=100000)
    model.fit(X_train, y_train)

    return model

# Train the model on both merged training datasets
model_image01 = train_lasso_classifier(train_df_image01)
model_image01_sliding = train_lasso_classifier(train_df_image01_sliding)

# Paths to test datasets for prediction and evaluation
test_paths = {
    "image0": "output_2/image0_test.csv",
    "image0_sliding": "output_2/image0_sliding_test.csv",
    "image01": "output_2/image01_test.csv",
    "image01_sliding": "output_2/image01_sliding_test.csv"
}

# Output directory for predictions and confusion matrices
output_folder = "output_2"
os.makedirs(output_folder, exist_ok=True)

# Function to evaluate and save predictions with confusion matrices
def evaluate_model(model, test_df, category_name):
    X_test = test_df.drop(columns=['Label'])
    y_true = test_df['Label']
    y_pred = np.round(model.predict(X_test))  # Round for binary classification

    # Add predicted labels to the test DataFrame
    test_df['Predicted_Label'] = y_pred
    test_df.to_csv(f"{output_folder}/{category_name}_test_with_predictions.csv", index=False)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix for {category_name}:\n{cm}")
    
    # Save confusion matrix
    np.savetxt(f"{output_folder}/{category_name}_confusion_matrix.csv", cm, delimiter=",", fmt='%d')
    
    # Calculate and return accuracy and F1 score
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, f1

# Evaluate models and store results
results = {}

# Evaluate on all test sets using the model trained on merged datasets
for name, path in test_paths.items():
    test_df = pd.read_csv(path)
    model = model_image01 if "sliding" not in name else model_image01_sliding
    accuracy, f1 = evaluate_model(model, test_df, name)
    results[name] = {"Accuracy": accuracy, "F1 Score": f1}

# Print results for comparison
print("\nModel Performance Summary:")
for category, metrics in results.items():
    print(f"{category} - Accuracy: {metrics['Accuracy']:.4f}, F1 Score: {metrics['F1 Score']:.4f}")
    
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os

# Add paths for image012 and image012_sliding
file_paths.update({
    "image012": "output/image012.csv",
    "image012_sliding": "output/image012_sliding.csv"
})

# Process image012 and image012_sliding like other datasets
for name, path in file_paths.items():
    df = pd.read_csv(path)
    
    # Split each dataset into training and testing sets
    train_df, test_df = split_dataset(df)
    
    # Save the results in the output folder
    train_df.to_csv(f"{output_folder}/{name}_train.csv", index=False)
    test_df.to_csv(f"{output_folder}/{name}_test.csv", index=False)


# Adjusted paths for training and corresponding test datasets (direct mapping)
train_test_mapping = {
    "image01": "output_2/image01_test.csv",
    "image01_sliding": "output_2/image01_sliding_test.csv",
    "image012": "output_2/image012_test.csv",
    "image012_sliding": "output_2/image012_sliding_test.csv"
}

# Output directory for saving predictions and confusion matrices
output_folder = "output_2"
os.makedirs(output_folder, exist_ok=True)

# Function to load and sample large datasets
def load_and_sample_data(path, max_samples=1000000):
    df = pd.read_csv(path)
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
    X = df.drop(columns=['Label'])
    y = df['Label']
    return X, y

# Function to train Random Forest model
def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

# Function to evaluate the model and save results
def evaluate_model(model, X_test, y_true, category_name):
    y_pred = model.predict(X_test)

    # Save predicted labels to test DataFrame
    test_results = pd.DataFrame({"Actual_Label": y_true, "Predicted_Label": y_pred})
    test_results.to_csv(f"{output_folder}/{category_name}_predictions_rf.csv", index=False)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"{category_name} - Confusion Matrix:\n{cm}\n")
    
    # Save confusion matrix
    np.savetxt(f"{output_folder}/{category_name}_confusion_matrix_rf.csv", cm, delimiter=",", fmt='%d')
    
    # Calculate accuracy and F1 score
    accuracy = accuracy_score(y_true, y_pred)
    n_classes = len(np.unique(y_true))
    f1 = f1_score(y_true, y_pred, average="binary" if n_classes == 2 else "weighted")

    return accuracy, f1

# Dictionary to store results
results_rf = {}

# Train and evaluate Random Forest for each dataset category
for train_name, test_path in train_test_mapping.items():
    # Load training data using simplified file names
    X_train, y_train = load_and_sample_data(f"output_2/{train_name}_train.csv")
    n_classes = 3 if "012" in train_name else 2
    print(f"\nTraining Random Forest for {train_name} (Classes: {n_classes})")
    
    # Train model
    model_rf = train_random_forest(X_train, y_train)

    # Evaluate the model on its corresponding test set
    X_test, y_true = load_and_sample_data(test_path)
    accuracy, f1 = evaluate_model(model_rf, X_test, y_true, train_name)
    results_rf[train_name] = {"Accuracy": accuracy, "F1 Score": f1}

# Print model performance summary
print("\nSummary of Random Forest Model Performance:")
for category, metrics in results_rf.items():
    print(f"{category} - Accuracy: {metrics['Accuracy']:.4f}, F1 Score: {metrics['F1 Score']:.4f}")

# Task 4
# Paths to datasets for evaluation (assumes files are already split and trained)
train_test_mapping = {
    "image01": ("output_2/image01_train.csv", "output_2/image01_test.csv"),
    "image01_sliding": ("output_2/image01_sliding_train.csv", "output_2/image01_sliding_test.csv"),
    "image012": ("output_2/image012_train.csv", "output_2/image012_test.csv"),
    "image012_sliding": ("output_2/image012_sliding_train.csv", "output_2/image012_sliding_test.csv")
}

# Store results
results = {}

# Function to load data
def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=['Label'])
    y = df['Label']
    return X, y

# Task 4 - Evaluate both Lasso and Random Forest on each dataset
for dataset_name, (train_path, test_path) in train_test_mapping.items():
    # Load data
    X_train, y_train = load_data(train_path)
    X_test, y_true = load_data(test_path)
    n_classes = len(np.unique(y_train))

    # Initialize dictionary for dataset results
    results[dataset_name] = {}

    # Lasso Model (only on binary datasets)
    if n_classes == 2:
        lasso_model = LassoCV(cv=5, random_state=42, max_iter=100000)
        lasso_model.fit(X_train, y_train)
        y_pred_lasso = np.round(lasso_model.predict(X_test))
        accuracy_lasso = accuracy_score(y_true, y_pred_lasso)
        f1_lasso = f1_score(y_true, y_pred_lasso, average="binary")
        cm_lasso = confusion_matrix(y_true, y_pred_lasso)
        
        results[dataset_name]['Lasso'] = {
            "Accuracy": accuracy_lasso,
            "F1 Score": f1_lasso,
            "Confusion Matrix": cm_lasso
        }

    # Random Forest Model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    accuracy_rf = accuracy_score(y_true, y_pred_rf)
    f1_rf = f1_score(y_true, y_pred_rf, average="binary" if n_classes == 2 else "weighted")
    cm_rf = confusion_matrix(y_true, y_pred_rf)
    
    results[dataset_name]['Random Forest'] = {
        "Accuracy": accuracy_rf,
        "F1 Score": f1_rf,
        "Confusion Matrix": cm_rf
    }

# Displaying Results
for dataset, models in results.items():
    print(f"\nDataset: {dataset}")
    for model_name, metrics in models.items():
        print(f"Model: {model_name}")
        print(f"  Accuracy: {metrics['Accuracy']:.4f}")
        print(f"  F1 Score: {metrics['F1 Score']:.4f}")
        print(f"  Confusion Matrix:\n{metrics['Confusion Matrix']}\n")

# Comparison Analysis for Task 4
# Analyzing best-performing model and dataset combination based on Accuracy and F1 Score

# Initialize variables for best model and dataset
best_model = None
best_dataset = None
best_accuracy = 0
best_f1 = 0

for dataset, models in results.items():
    for model_name, metrics in models.items():
        # Compare accuracy and F1 score to determine the best-performing model-dataset pair
        if metrics['Accuracy'] > best_accuracy or (metrics['Accuracy'] == best_accuracy and metrics['F1 Score'] > best_f1):
            best_accuracy = metrics['Accuracy']
            best_f1 = metrics['F1 Score']
            best_model = model_name
            best_dataset = dataset

print("\nBest Model-Dataset Pair:")
print(f"Dataset: {best_dataset}")
print(f"Model: {best_model}")
print(f"  Accuracy: {best_accuracy:.4f}")
print(f"  F1 Score: {best_f1:.4f}")
