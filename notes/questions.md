# Questions

**Old**:

* Which GNN architecture should I benchmark ? use TSL with lstm, gru for temporal and GATConv, diffconv, graphconv for spatial
* What horizon and window should I use for benchmarking ? 24h history, 1 hour forecast, 1 week history, next day horizon
* Hyperparameter optimization: no more than two graph layers, temporal: 1, 3, 5, hidden size: power of two: 16, 32, 64, k = 4 (optuna)
* How sparse should be graph be ? Use quantile
* Memory error with lstm cell during training despite only ~300k parameters

![alt text](./images/image-8.png)

can drop but try to cmp with spatial block in UGNET

* selected sensors: currently, daily mean look OK (**show plots**) but not a simple plot, which shows bad results when forecasting 1 hour horizon. Should I restrict more the sensors for hourly horizon and work with daily means (i.e, change sampling period) for daily predicitions? OK

**New**:

* How to access variance in the prediction ?

try to add graph learning in ugnet
change block in ugnet one by one then compare
check number of epochs in diffstg code
adapt config to sampling rate sensors,..;
model evaluation:
IOT: electricty data: open dataset and no need to learn graph
