import pandas as pd
from chronos import Chronos2Pipeline
import torch
import numpy as np


# EWZ

df = pd.read_csv("../../data/EWZ_Daily_Preprocessed.csv")
df.rename(columns={"Name": "timestamp"}, inplace=True)


split_idx = int(len(df) * 0.85) +1

# Split the dataframe
train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

train_df.drop(columns='timestamp', inplace=True)
test_df.drop(columns='timestamp', inplace=True)



train_values = train_df.values

# mean_train = np.mean(train_values, axis=0)
# std_train = np.std(train_values, axis=0)

# train_values = (train_values - mean_train) / std_train

context = torch.tensor(train_values.T, dtype=torch.float32)
N, T = context.shape

context = context.reshape(N, 1, T)

forecast = np.array(pipeline.predict(
    inputs=context,
    prediction_length=len(test_df),
))



y_true = test_df.values

# y_true = (y_true - mean_train) / std_train

y_true = y_true.T

N, T = y_true.shape
y_true = y_true.reshape(N, 1, T)

np.save("../diffstg/preds/raw_ewz_daily_pred_chronos.npy", forecast)
np.save("../diffstg/preds/raw_ewz_daily_true_chronos.npy", y_true)


# Elergone


elec_data = np.load("../diffstg/data/dataset/electricity_benchmark/flow.npy")
elec_data = elec_data.astype(np.float64)
T, N, _ = elec_data.shape


split_idx = int(T * 0.85)
elec_data = np.transpose(elec_data, (1, 2, 0)) # N, 1, T

context = elec_data[:, :, :split_idx].astype(np.float64)

# train_mean = np.mean(context, keepdims=True, axis=2)
# std_train = np.std(context, keepdims=True, axis=2) + 1e-9

test = elec_data[:, :, split_idx:].astype(np.float64)



total_pred_length = np.shape(test)[2]
chunk_size = 512

# Initialize forecast array
forecast = np.zeros((N, 1, 21, total_pred_length))

# Start with the initial context
current_context = context.copy()

for step in range(0, total_pred_length, chunk_size):
    pred_len = min(chunk_size, total_pred_length - step)

    # Predict next chunk for ALL time series at once
    batch_forecast = np.array(
        pipeline.predict(
        inputs=current_context,
        prediction_length=pred_len,
    )
    )

    # print(np.array(batch_forecast).shape, forecast.shape)

    # Take mean across samples (probabilistic forecasting)
    # batch_forecast = np.mean(np.array(batch_forecast), axis=2)

    # Store predictions
    forecast[:, :,:, step:step+pred_len] = batch_forecast

    # Update context for next iteration (append predictions)
    current_context = np.concatenate([
        current_context,
        np.mean(batch_forecast, axis=2)
    ], axis=2)

    print(f"Completed {step + pred_len}/{total_pred_length} steps")

    torch.cuda.empty_cache()


forecast = forecast[:, :, :, :total_pred_length]



np.save("../diffstg/preds/raw_electricity_benchmark_pred_chronos.npy", forecast)
np.save("../diffstg/preds/raw_electricity_benchmark_true_chronos.npy", test)

