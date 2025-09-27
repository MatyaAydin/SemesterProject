# Questions
**Old**:
* Which GNN architecture should I benchmark ? use TSL with lstm, gru for temporal and GATConv, diffconv, graphconv for spatial
* Do I have access to compute resources ? See later, should not need until diffusion
* anomaly detection: how? must it be the same architecture as the one for forecasting ? See later

**New**:
* What horizon and window should I use for benchmarking ?
* Hyperparameter optimization: optuna ? how many layers, neighbors, hidden size, kernel size, activation functions
* drop given name (boilers) or manually: 30-40: do passes and be more strict


remove for train not test to detect outliers
median filtering: take window of 6, 8: just for outlier or smooth whole time series ?

horizon: 24 hours, 1 hour

24h history, 1 hour forecast
1 week history, next day horizon

no more than two graph layers
temporal: 1, 3, 5
hidden size: power of two: 16, 32, 64
k = 4

optuna


