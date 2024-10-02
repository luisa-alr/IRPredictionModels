## Machine Learning in Finance - Interest Rate Prediction
Luisa Rosa - Fall 2024
Interest Rate Prediction using Vasicek and CIR models

---
## Instructions:
+ Download all files (1 Python program and 1 CSV dataset)
+ Run Q4.py
+ Plots will be saved
+ To see the answers to the questions, they will printed out to the terminal but are also included in the pdf

---
## Question 4: 
The data file ”IR-data.xlsx” contains approximately 11 years of weekly interest rate data. Use the data points prior to 2021 to calibrate the Vasicek and CIR models, respectively.
Use the calibrated models to predict the interest rate for each test data point in 2021 and beyond. For each model, report the following model parameters and outcomes:

*   a) a, b, and σ

*   b) For each predicted point, report the forecasted value and associated volatility (i.e., standard deviation)

*   c) The overall MSE and R2 for the test data

### Solution:
Vasicek and CIR model predictions.

---

*  a) Vasicek optimized parameters: a = 0.0018759731562042567, b = 2.5811470673215773, sigma = 0.056765551548106426
      CIR optimized parameters: a = 0.0018759731562042418, b = 2.581146487427231, sigma = 0.05676555154810597


*  b) Output of the following table for the test data:
      Date  Actual_IR  Predicted_IR_Vasicek  Predicted_IR_CIR  Vasicek_volatility  CIR_volatility

      Vasicek overall volatility: 0.056867995872713725
      CIR overall volatility:0.05868268979485365


*  c) Vasicek Test MSE: 3.5060524785551763
      Vasicek Test R²: -1.517266789123049
      CIR Test MSE: 2.0986433299651184
      CIR Test R²: -0.506778688866867

