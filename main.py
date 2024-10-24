# Import necessary libraries
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
%matplotlib inline

# Load the dataset
df = pd.read_csv("D:\Madhav's Projects\Customer Churn Dataset\Customer_churn_dataset.csv")

# Preview 5 random samples from the dataset
df.sample(5)

# Drop 'customerID' column as it is not useful for model training
df.drop('customerID', axis='columns', inplace=True)

# Check the data types of each column
df.dtypes

# Check for invalid values in 'TotalCharges' column (non-numeric)
pd.to_numeric(df.TotalCharges, errors="coerce").isnull()

# Find rows where 'TotalCharges' have invalid (non-numeric) values
df[pd.to_numeric(df.TotalCharges, errors="coerce").isnull()]

# Filter out rows with invalid 'TotalCharges' values (i.e., empty strings)
df1 = df[df.TotalCharges != " "]

# Check the shape of the cleaned dataset
df1.shape

# Convert 'TotalCharges' from string to numeric
df1.TotalCharges = pd.to_numeric(df1.TotalCharges)

# Separate the tenure data based on churn status for plotting
tenure_churn_no = df1[df1.Churn == "No"].tenure
tenure_churn_yes = df1[df1.Churn == "Yes"].tenure

# Plot histogram to visualize the tenure distribution based on churn
plt.xlabel("Tenure")
plt.ylabel("Number of Customers")
plt.title("Customer Churn Prediction")

plt.hist([tenure_churn_yes, tenure_churn_no], color=['green', 'red'], label=("Churn=Yes", "Churn=No"))
plt.legend()

# Function to print unique values of categorical columns
def print_unique_col_values(df):
    for column in df:
        if df[column].dtypes == 'object':
            print(f"{column} : {df[column].unique()}")

# Print unique values of categorical columns
print_unique_col_values(df1)

# Replace 'No internet service' and 'No phone service' with 'No' for consistency
df1.replace("No internet service", "No", inplace=True)
df1.replace("No phone service", "No", inplace=True)

# Print unique values again after replacement
print_unique_col_values(df1)

# Convert Yes/No columns to 1/0
yes_no_columns = ["Partner", "Dependents", "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                  "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling", "Churn"]

for col in yes_no_columns:
    df1[col].replace({"Yes": 1, "No": 0}, inplace=True)

# Convert 'gender' column to numeric: Male -> 0, Female -> 1
df1['gender'].replace({"Male": 0, "Female": 1}, inplace=True)

# Create dummy variables for categorical columns
df2 = pd.get_dummies(data=df1, columns=["InternetService", "Contract", "PaymentMethod"])

# Check the new column names after creating dummy variables
df2.columns

# Preview 4 random samples from the updated dataset
df2.sample(4)

# Check data types after transformation
df2.dtypes

# Columns to be scaled
col_to_scale = ["tenure", "MonthlyCharges", "TotalCharges"]

# Apply MinMaxScaler to scale the selected columns
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

df2[col_to_scale] = scaler.fit_transform(df2[col_to_scale])

# Check unique values of each column to ensure correct scaling
for col in df2:
    print(f" {col} : {df2[col].unique()} ")

# Split the dataset into input (X) and output (y)
x = df2.drop("Churn", axis="columns")
y = df2["Churn"]

# Split the data into training and testing sets (80% train, 20% test)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=5)

# Build a neural network model using TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras

# Define the model architecture
model = keras.Sequential([
    keras.layers.Dense(26, input_shape=(26,), activation="relu"),  # Input layer and hidden layer
    keras.layers.Dense(1, activation="sigmoid")  # Output layer
])

# Compile the model with binary crossentropy loss and Adam optimizer
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model on the training set
model.fit(X_train, Y_train, epochs=100)

# Evaluate the model on the test set
model.evaluate(X_test, Y_test)

# Predict churn on the test set
yp = model.predict(X_test)
yp[:5]

# Convert the predictions into binary values (1 for churn, 0 for no churn)
y_pred = []
for element in yp:
    if element > 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)

# Import metrics for evaluation
from sklearn.metrics import confusion_matrix, classification_report

# Print classification report to evaluate model performance
print(classification_report(Y_test, y_pred))

# Plot confusion matrix to visualize the performance
import seaborn as sn
cm = tf.math.confusion_matrix(labels=Y_test, predictions=y_pred)

# Plot the confusion matrix using a heatmap
plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Prediction")
plt.ylabel("Truth")
