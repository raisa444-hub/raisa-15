# raisa-15
Advance timeseries forecasting using neural network and explainable AI
description
This project develops, trains and evaluates deep learning models (LSTM or TCN) for multivariate time series forecasting and compares them to a classical statistical baseline (e.g., SARIMAX). It includes a robust preprocessing pipeline (scaling, missing-value imputation, feature engineering), unit tests for the pipeline, and an explainability stage using SHAP (or similar) to quantify feature importance for model predictions. Deliverables include runnable Python code, saved model files and a written report summarizing dataset characteristics, modeling choices, hyperparameters and XAI findings.
Methodology
Methodology (step-by-step)
Data acquisition & inspection
Load the multivariate time series CSV (timestamp + features + target(s)).
Visualize series, check missing values, correlations and stationarity.
Preprocessing pipeline
Fill missing values (forward/backward or interpolation).
Create time features (hour, day-of-week, month) if appropriate.
Scale features (fit scalers on training set only).
Save pipeline and provide unit tests for imputation and scaling correctness.
Windowing (sequence creation)
Convert series into supervised windows: input sequences of length T_in predict T_out steps ahead (here we'll use 1-step ahead by default).
Shuffle train windows; preserve temporal order for validation/test.
Baselines
Fit a classical statistical baseline (ARIMA/SARIMAX) on each target or on aggregated series for comparison.
Deep model
Build LSTM model (TensorFlow / Keras) with configurable layers, dropout, and learning schedule.
Use callbacks (ModelCheckpoint, EarlyStopping).
Training and evaluation
Train on train set, evaluate on test set with RMSE, MAE, plot predictions vs ground truth.
Save plots and numeric results.
Explainability (XAI)
Use SHAP (GradientExplainer or KernelExplainer) for feature contribution on a sample of test windows.
Produce summary plots and per-prediction explanations (feature importance).
Reporting
Save a textual report (dataset summary, hyperparameters, metrics, XAI insights) and include saved model and unit test outputs.
