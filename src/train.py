# import pandas as pd


# # Load training data
# train_data = pd.read_csv("data/training_data_final.csv")

# print(train_data.head())
# print("\nColumns:\n", train_data.columns)
# print("\nInfo:\n")
# print(train_data.info())


import pandas as pd
import joblib # to save the model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error



# Load data
train_data = pd.read_csv("data/training_data_final.csv")

# Drop useless column
train_data = train_data.drop(columns=["id_semester_evaluation"])

# Define features (X) and target (y)
X = train_data.drop(columns=["semester_evaluation_gtu_mark"])
y = train_data["semester_evaluation_gtu_mark"]

# Train model
model = LinearRegression()
model.fit(X, y)

print("Model trained successfully ✅")

# Load testing data
test_data = pd.read_csv("data/testing_data_final.csv")

# Drop ID column
test_data = test_data.drop(columns=["id_semester_evaluation"])

# Split features and target
X_test = test_data.drop(columns=["semester_evaluation_gtu_mark"])
y_test = test_data["semester_evaluation_gtu_mark"]

# Predict
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print("\nMean Absolute Error:", mae)

joblib.dump(model, "models/model.pkl")

print("\nModel saved to models/model.pkl ✅")

# Print predictions vs actual
print("\nPredictions vs Actual:\n")
for i in range(5):
    print(f"Predicted: {y_pred[i]:.2f}, Actual: {y_test.iloc[i]}")