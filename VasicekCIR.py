# Luisa Rosa
# HW#1 - Machine Learning in Finance
# Question 4

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, r2_score


def cleanData(df):
    df["DATE"] = pd.to_datetime(df["DATE"])  # fix data datetime format
    df["IR"] = pd.to_numeric(df["IR"], errors="coerce")  # fix missing numerical values
    df.dropna(subset=["IR"], inplace=True)  # drop rows with missing numerical values

    # Separate train and test datasets
    train_data = df[df["DATE"] < "2021-01-01"]
    test_data = df[df["DATE"] >= "2021-01-01"]

    return train_data, test_data


# Function to define Vasicek model
def vasicek_model(data, params, init):
    a, b, sigma = params
    dt = 1
    predictions = [init]
    for i in range(1, len(data)):
        r_prev = predictions[-1]
        r_next = (
            r_prev + a * (b - r_prev) * dt + sigma * np.sqrt(dt) * np.random.normal()
        )
        predictions.append(r_next)
    return np.array(predictions)


# Function to return the mse between the Vasicek model's predictions and the actual training data
def vasicek_mse(params, train_data):
    predictions = vasicek_model(train_data, params, train_data["IR"].iloc[0])
    return mean_squared_error(train_data["IR"], predictions)


# Function to define CIR model
def cir_model(data, params, init):
    a, b, sigma = params
    dt = 1
    predictions = [init]
    for i in range(1, len(data)):
        r_prev = predictions[-1]
        # Ensure that the previous rate is non-negative
        if r_prev < 0:
            r_prev = 0
        r_next = (
            r_prev + a * (b - r_prev) * dt + sigma * np.sqrt(r_prev) * np.random.normal()
        )
        predictions.append(r_next)
    return np.array(predictions)


# Function to return the mse between the CIR model's predictions and the actual training data
def cir_mse(params, train_data):
    predictions = cir_model(train_data, params, train_data["IR"].iloc[0])
    return mean_squared_error(train_data["IR"], predictions)

# Load data
df = pd.read_csv("DGS10.csv")

# Clean data
train_data, test_data = cleanData(df)

# Plot the data (Training and Testing IR)
plt.figure(figsize=(10, 6))
plt.plot(train_data["DATE"], train_data["IR"], label="Interest Rate")
plt.plot(test_data["DATE"], test_data["IR"], label="Interest Rate", color="red")
plt.xlabel("Date")
plt.ylabel("Interest Rate")
plt.title("Interest Rates (Training Data and Testing Data)")
plt.tight_layout()
plt.savefig("TrainingANDTestIR.jpg", format='jpg', dpi=300, bbox_inches='tight')
plt.show()


# Calibrate the models (train)
# Initial guess for a, b, sigma
# a = -np.log(train_data['IR'].autocorr(lag=1))
a = 0.1
b = train_data['IR'].mean()
sigma = np.std(np.diff(train_data['IR']))

params = a, b, sigma

# Train - Optimize the parameters to fit the model to the training data
v_result = minimize(
    vasicek_mse,
    params,
    args=(train_data,),
    method="L-BFGS-B",
    bounds=[(0, None), (None, None), (0, None)],
)
c_result = minimize(
    cir_mse,
    params,
    args=(train_data,),
    method="L-BFGS-B",
    bounds=[(0, None), (None, None), (0, None)],
)

# A) Estimate parameters 𝑎, 𝑏, and 𝜎, extracting the optimized parameters
print("QUESTION A\n")
a_opt, b_opt, sigma_opt = v_result.x
print(f"Vasicek optimized parameters: a = {a_opt}, b = {b_opt}, sigma = {sigma_opt}")
a_opt, b_opt, sigma_opt = c_result.x
print(f"CIR optimized parameters: a = {a_opt}, b = {b_opt}, sigma = {sigma_opt}\n")

# Test - Make predictions
v_test_predictions = vasicek_model(test_data, v_result.x, train_data["IR"].iloc[-1])
c_test_predictions = vasicek_model(test_data, c_result.x, train_data["IR"].iloc[-1])

v_mse = mean_squared_error(test_data["IR"], v_test_predictions)
v_r2 = r2_score(test_data["IR"], v_test_predictions)
c_mse = mean_squared_error(test_data["IR"], c_test_predictions)
c_r2 = r2_score(test_data["IR"], c_test_predictions)

# B) Report the forecasted value and associated volatility
print("QUESTION B\n")
report_df = pd.DataFrame(
    {
        "Date": test_data["DATE"],
        "Actual_IR": test_data["IR"],
        "Predicted_IR_Vasicek": v_test_predictions,
        "Predicted_IR_CIR": c_test_predictions,
    }
)

# Compute volatility for each date as the absolute value of daily changes
report_df["Vasicek_volatility"] = np.abs(
    report_df["Actual_IR"] - report_df["Predicted_IR_Vasicek"]
)
report_df["CIR_volatility"] = np.abs(
    report_df["Actual_IR"] - report_df["Predicted_IR_CIR"]
)

v_overall_volatility = np.std(np.diff(v_test_predictions))
c_overall_volatility = np.std(np.diff(c_test_predictions))

print(report_df.to_string(index=False))

print(f"\nVasicek overall volatility: {v_overall_volatility}")
print(f"CIR overall volatility:{c_overall_volatility}\n")


# Plot the test predictions vs actual values
plt.figure(figsize=(10, 6))
plt.plot(test_data["DATE"], test_data["IR"], label="Actual Interest Rate", color="black")
plt.plot(test_data["DATE"], v_test_predictions, label="Vasicek Predicted Interest Rate", color="red")
plt.plot(test_data["DATE"], c_test_predictions, label="CIR Predicted Interest Rate", color="blue")
plt.xlabel("Date")
plt.ylabel("Interest Rate")
plt.title("Actual vs Predicted Interest Rates (Test Data)")
plt.legend()
plt.tight_layout()
plt.savefig("ModelsIR.jpg", format='jpg', dpi=300, bbox_inches='tight')
plt.show()

# C) Compare prediction with actual values. The overall MSE and R2 for the test data
print("QUESTION C\n")
print(f"Vasicek Test MSE: {v_mse}")
print(f"Vasicek Test R²: {v_r2}")
print(f"CIR Test MSE: {c_mse}")
print(f"CIR Test R²: {c_r2}")

