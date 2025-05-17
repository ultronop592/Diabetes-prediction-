# Diabetes-prediction-

Project Overview

This project, titled Diabetes Prediction Using ML Algorithms, aims to predict whether a person has diabetes based on health-related features like glucose levels, BMI, and age. It uses machine learning (ML) techniques, specifically the Support Vector Machine (SVM) algorithm, to build a prediction model. The project is implemented in Python using a Jupyter Notebook and includes data analysis, preprocessing, model training, evaluation, and a prediction system.
Objectives

Build a machine learning model to predict diabetes (diabetic or non-diabetic).
Analyze the dataset to understand patterns and relationships.
Evaluate the model's performance using accuracy metrics.
Create a system to make predictions for new data.

Dataset
The dataset used is diabetes.csv, which contains health data for 768 patients. It has 9 columns:

Pregnancies: Number of times pregnant
Glucose: Blood glucose level
BloodPressure: Blood pressure measurement
SkinThickness: Thickness of skin fold
Insulin: Insulin level in blood
BMI: Body Mass Index
DiabetesPedigreeFunction: A measure of diabetes likelihood based on family history
Age: Age of the patient
Outcome: Target variable (0 = non-diabetic, 1 = diabetic)

Dataset Insights

Total rows: 768
Total columns: 9
Outcome distribution: 500 non-diabetic (0), 268 diabetic (1)
Some features (e.g., Glucose, BloodPressure) have minimum values of 0, which may indicate missing or invalid data.

Project Steps
1. Importing Dependencies
The project uses the following Python libraries:

pandas and numpy for data manipulation
sklearn.preprocessing.StandardScaler for data standardization
sklearn.model_selection.train_test_split for splitting data into training and testing sets
sklearn.svm.SVC for the SVM model
sklearn.metrics.accuracy_score for evaluating model performance

2. Data Loading and Exploration

The dataset is loaded using pandas.read_csv() from the file diabetes.csv.
Initial exploration includes:
Displaying the first 5 rows with diabetes_dataset.head().
Checking the dataset size with diabetes_dataset.shape (768 rows, 9 columns).
Statistical summary using diabetes_dataset.describe() to understand data distribution (e.g., mean, min, max).
Counting outcomes with diabetes_dataset['Outcome'].value_counts() to check class balance (500 non-diabetic, 268 diabetic).



3. Data Preprocessing

Separating Features and Target: Features (X) are all columns except Outcome, and the target (y) is the Outcome column.
Standardization: Features are standardized using StandardScaler to ensure all features have a mean of 0 and a standard deviation of 1, which improves model performance.

4. Train-Test Split

The dataset is split into training (80%) and testing (20%) sets using train_test_split.
Training set: 614 rows
Testing set: 154 rows


The split is stratified (stratify=y) to maintain the proportion of diabetic and non-diabetic cases in both sets.

5. Model Training

A Support Vector Machine (SVM) classifier with a linear kernel is used (svm.SVC(kernel='linear')).
The model is trained on the standardized training data (x_train, y_train).

6. Model Evaluation

Training Accuracy: The model’s accuracy on the training data is 78.34%.
Testing Accuracy: The model’s accuracy on the testing data is 77.27%.
The similar accuracies suggest the model generalizes well and is not overfitting.

7. Making Predictions

A prediction system is built to classify new data:
Input data (e.g., (5, 116, 74, 0, 0, 25.6, 0.201, 30)) is converted to a NumPy array and reshaped.
The input is standardized using the same StandardScaler.
The trained SVM model predicts the outcome.
For the example input, the prediction is 0 (non-diabetic), and a message is printed: "the person is not diabetic".



Results

Model Performance:
Training Accuracy: 78.34%
Testing Accuracy: 77.27%


Prediction Example:
Input: (5, 116, 74, 0, 0, 25.6, 0.201, 30)
Output: Non-diabetic (0)



