# Task-2: Model Evaluation and Comparision
# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from google.colab import files

# Step 2: Upload the dataset (Manually upload the dataset file)
print("Please upload your dataset file (.csv) using the button below.")
uploaded = files.upload()

# Get the filename of the uploaded dataset
filename = next(iter(uploaded))

# Step 3: Load the dataset into a pandas DataFrame
data = pd.read_csv(filename)

# Print dataset overview and column names
print("Dataset Overview:")
print(data.head())
print("Columns in dataset:", data.columns)

# Remove leading/trailing spaces in column names (if any)
data.columns = data.columns.str.strip()

# Print column names after stripping spaces
print("Cleaned Columns in dataset:", data.columns)

# Step 4: Handle Missing Values
# Impute numerical columns with the mean
numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
print(f"Numerical Columns to Impute: {numerical_columns}")
imputer = SimpleImputer(strategy='mean')
data[numerical_columns] = imputer.fit_transform(data[numerical_columns])

# Impute categorical columns with the most frequent value (mode)
categorical_columns = data.select_dtypes(include=['object']).columns
print(f"Categorical Columns to Impute: {categorical_columns}")
imputer_cat = SimpleImputer(strategy='most_frequent')
data[categorical_columns] = imputer_cat.fit_transform(data[categorical_columns])

# Step 5: One-Hot Encoding for Categorical Variables
print("Performing One-Hot Encoding on Categorical Variables...")
data = pd.get_dummies(data, drop_first=True)

# Check the dataset after One-Hot Encoding
print("Data after One-Hot Encoding:")
print(data.head())

# Step 6: Scale Numerical Features
print("Scaling Numerical Features...")
scaler = StandardScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Step 7: Split Data into Features and Target
# Update the target column name to the new one after one-hot encoding
target_column = 'Target_Yes'  # The target column after one-hot encoding

# Check if the target column exists in the dataset
if target_column not in data.columns:
    print(f"Warning: '{target_column}' not found in dataset columns.")
    print("Please check the column name and adjust the code.")
else:
    # Split the dataset into features (X) and target (y)
    X = data.drop(target_column, axis=1)  # Features (everything except the target column)
    y = data[target_column]  # Target

    # Check the shape of X and y to ensure they match
    print(f"Features (X) shape: {X.shape}, Target (y) shape: {y.shape}")

    # Step 8: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Final confirmation
    print("Data processing complete!")
    print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

    # Step 9: Train and Evaluate Models
    def evaluate_model(model, X_train, X_test, y_train, y_test):
        model.fit(X_train, y_train)  # Train the model
        y_pred = model.predict(X_test)  # Make predictions

        # Evaluate the model using different metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Print evaluation results
        print(f"Model: {model.__class__.__name__}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print('-'*30)

    # Initialize models
    logistic_model = LogisticRegression(max_iter=1000, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Evaluate each model
    evaluate_model(logistic_model, X_train, X_test, y_train, y_test)
    evaluate_model(rf_model, X_train, X_test, y_train, y_test)


