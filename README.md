## Machine Learning in Finance - Interest Rate Prediction
Luisa Rosa - Fall 2024
Interest Rate Prediction using Vasicek, CIR, RNN, LSTM, and Transformer models

---
## Instructions:
+ Download all files (3 Python programs and 1 CSV dataset)
+ Run Q4.py
+ Run Q6.py
+ Run Q7_extra.py
+ Plots will be saved
+ To see the answers to the questions, they will printed out to the terminal but are also included in the pdf

---
## Question 4: 
The data file ”IR-data.xlsx” contains approximately 11 years of weekly interest rate data. Use the data points prior to 2021 to calibrate the Vasicek and CIR models, respectively.
Use the calibrated models to predict the interest rate for each test data point in 2021 and beyond. For each model, report the following model parameters and outcomes:

*   a) a, b, and σ

*   b) For each predicted point, report the forecasted value and associated volatility (i.e., standard deviation)

*   c) The overall MSE and R2 for the test data

## Question 6:
The data file ”DGS10.xlsx” contains approximately 11 years of weekly interest rate
data. Use the data points prior to 2021 to train RNN and LSTM models, respectively, using
50 data points as the lag window. Use the trained models to predict the interest rate for each
test data point in 2021 and beyond. For each model, report the following model parameters and
outcome:
(a) The model architecture summary output
(b) # of epochs trained and final training MSE loss
(c) For each predicted point, report the forecasted value
(d) The overall MSE and R2 for the test data

## Question 7:
Repeat Q4/Q6 using a Transformer model. Keep the lag window size the
same and report the same model parameters and outcome.

### Solution:
Vasicek, CIR, RNN, LSTM, and Transformer model predictions.

---

*  a) Vasicek optimized parameters: a = 0.0018759731562042567, b = 2.5811470673215773, sigma = 0.056765551548106426
  
      CIR optimized parameters: a = 0.0018759731562042418, b = 2.581146487427231, sigma = 0.05676555154810597


*  b) Output of the following table for the test data:
  
      Date | Actual_IR | Predicted_IR_Vasicek | Predicted_IR_CIR | Vasicek_volatility | CIR_volatility

      Vasicek overall volatility: 0.056867995872713725

      CIR overall volatility:0.05868268979485365


*  c) Vasicek Test MSE: 3.5060524785551763
  
      Vasicek Test R²: -1.517266789123049

      CIR Test MSE: 2.0986433299651184

      CIR Test R²: -0.506778688866867

*  RNN model:
Model Architecture Summary:
Model: "sequential_6"
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ simple_rnn_3 (SimpleRNN)             │ (None, 100)                 │          10,200 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_6 (Dense)                      │ (None, 1)                   │             101 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
```
 Total params: 20,604 (80.49 KB)
 Trainable params: 10,301 (40.24 KB)
 Non-trainable params: 0 (0.00 B)
 Optimizer params: 10,303 (40.25 KB)

Number of Epochs Trained: 71
Final Training MSE Loss: 0.00017410848522558808
Test Data MSE: 0.005280131687810965
Test Data R²: 0.9958597715397697

*  LSTM model:
Model Architecture Summary:
Model: "sequential_7"
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ lstm_3 (LSTM)                        │ (None, 100)                 │          40,800 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_7 (Dense)                      │ (None, 1)                   │             101 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
```
 Total params: 81,804 (319.55 KB)
 Trainable params: 40,901 (159.77 KB)
 Non-trainable params: 0 (0.00 B)
 Optimizer params: 40,903 (159.78 KB)

Number of Epochs Trained: 73
Final Training MSE Loss: 0.00020874357142020017
Test Data MSE: 0.005621806049031487
Test Data R²: 0.995591859677321

*  Transformer:
Model Architecture Summary:
Model: "functional_15"
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)              ┃ Output Shape           ┃        Param # ┃ Connected to           ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ input_layer_18            │ (None, 50, 1)          │              0 │ -                      │
│ (InputLayer)              │                        │                │                        │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ dense_43 (Dense)          │ (None, 50, 32)         │             64 │ input_layer_18[0][0]   │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ positional_encoding_laye… │ (None, 50, 32)         │              0 │ dense_43[0][0]         │
│ (PositionalEncodingLayer) │                        │                │                        │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ multi_head_attention_30   │ (None, 50, 32)         │          8,416 │ positional_encoding_l… │
│ (MultiHeadAttention)      │                        │                │ positional_encoding_l… │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ layer_normalization_30    │ (None, 50, 32)         │             64 │ multi_head_attention_… │
│ (LayerNormalization)      │                        │                │                        │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ dropout_76 (Dropout)      │ (None, 50, 32)         │              0 │ layer_normalization_3… │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ multi_head_attention_31   │ (None, 50, 32)         │          8,416 │ dropout_76[0][0],      │
│ (MultiHeadAttention)      │                        │                │ dropout_76[0][0]       │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ layer_normalization_31    │ (None, 50, 32)         │             64 │ multi_head_attention_… │
│ (LayerNormalization)      │                        │                │                        │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ dropout_78 (Dropout)      │ (None, 50, 32)         │              0 │ layer_normalization_3… │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ flatten_15 (Flatten)      │ (None, 1600)           │              0 │ dropout_78[0][0]       │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ dense_44 (Dense)          │ (None, 50)             │         80,050 │ flatten_15[0][0]       │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ dropout_79 (Dropout)      │ (None, 50)             │              0 │ dense_44[0][0]         │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ dense_45 (Dense)          │ (None, 1)              │             51 │ dropout_79[0][0]       │
└───────────────────────────┴────────────────────────┴────────────────┴────────────────────────┘
```
 Total params: 291,377 (1.11 MB)
 Trainable params: 97,125 (379.39 KB)
 Non-trainable params: 0 (0.00 B)
 Optimizer params: 194,252 (758.80 KB)

Number of Epochs Trained: 36
Final Training MSE Loss: 0.0018350633326917887
Test Data MSE: 0.00888606417485663
Test Data R²: 0.9930323071522812


#### Please note that the complete list of each predicted point and the forecasted value can be found on the output or the Written answers pdf. It is not added here as it is lenghty.
