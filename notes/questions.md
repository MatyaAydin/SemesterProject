# Questions

**Old**:

* Which GNN architecture should I benchmark ? use TSL with lstm, gru for temporal and GATConv, diffconv, graphconv for spatial
* What horizon and window should I use for benchmarking ? 24h history, 1 hour forecast, 1 week history, next day horizon
* Hyperparameter optimization: no more than two graph layers, temporal: 1, 3, 5, hidden size: power of two: 16, 32, 64, k = 4 (optuna)
* How sparse should be graph be ? Use quantile
* Memory error with lstm cell during training despite only ~300k parameters

![alt text](./images/image-8.png)

can drop but try to cmp with spatial block in UGNET

**New**:

* How to access variance in the prediction ? USE N_sample
* Preds seem to overshoot. Might be because Outliers are not removed in ewz_preprocessed. Ask for script so I can merge all preprocessing in one.
* try to add graph learning in ugnet: show where in code. Use adj.npy normalized as prior (+ Laplace smoothing to get nonzero proba) ?

ablation study: comapre with vs without graph learning, and other blocks and eletrical dataset

anomaly detection: build many c% CI and classify anomalies

change block in ugnet one by one then compare
IOT: electricty data: open dataset and no need to learn graph (just to benchmark) use get_connectivity from tsl and compare with knn lernable
