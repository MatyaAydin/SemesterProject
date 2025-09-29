# Questions
**Old**:
* Which GNN architecture should I benchmark ? use TSL with lstm, gru for temporal and GATConv, diffconv, graphconv for spatial
* Do I have access to compute resources ? See later, should not need until diffusion
* anomaly detection: how? must it be the same architecture as the one for forecasting ? See later
* What horizon and window should I use for benchmarking ? 24h history, 1 hour forecast, 1 week history, next day horizon
* Hyperparameter optimization: no more than two graph layers, temporal: 1, 3, 5, hidden size: power of two: 16, 32, 64, k = 4 (optuna)

**New**:
* Should median filter be causal ? do I apply it on the whole time series or just the outliers ?







