# Questions

**Old**:

* Which GNN architecture should I benchmark ? use TSL with lstm, gru for temporal and GATConv, diffconv, graphconv for spatial
* What horizon and window should I use for benchmarking ? 24h history, 1 hour forecast, 1 week history, next day horizon
* Hyperparameter optimization: no more than two graph layers, temporal: 1, 3, 5, hidden size: power of two: 16, 32, 64, k = 4 (optuna)
* How sparse should be graph be ? Use quantile
* Memory error with lstm cell during training despite only ~300k parameters
* Use adj.npy normalized as prior (+ Laplace smoothing to get nonzero proba) ? --> try init if time

* Electricitybenchmark: no get_connectivity. Try other tsl dataset or do same trick with covariance ? use elergon
* conservation laws ? source node; sum of neighbors = source [OPTIONAL, does not work]

![alt text](./images/image-8.png)

can drop but try to cmp with spatial block in UGNET

**New**:

* Ask opinion on method for anomaly detection (posteriori) in time series with diffusion model, if irrelevant will do more usual thing. How to assess quality ? (use def to get supervised dataset ?) no label, just show on real data
can add anomaly in random time interval, drift fault
current ideas:
    * pred is outside mean +- std DONE
    * error too large
    * compare with most correlated sensors

Done this since last meeting:
all on ewz_daily (much shorter training time)

* preds on Chronos
* retrained with 8 and 16 neighbors
* retrained with GRU instead of causal convolution
* show results: looks bad, do we just go with diffconv ? or ask to take a look at code after I clean to be sure everything looks correct ?

questions:

* Chronos trained on train + val: unfair comparison ? If I retrain on train + val after I can't do early stopping: dont use validation, fix epochs
* should context in chronos be T_history or whole df ?
* train on ewz daily preprocessed is really fast --> optuna ? but is it fair comparison ?
* train on one of the dataset from diffstg repo to be sure that there is no problem ?

Next steps:

* try pems08 with vanilla baseline and CSDI to check --> dimension prb
* retrain on electricity benchmark once I'm sure about what I'm doing
* relevant to try others graph convolution/temporal blocks?

* retrain with larger horizon
* anomaly detection: case study at the end, first show that forecasting works


organisation:
* Push deadline to 9th of january ?
* when do you go on vacation ?

new things I tried/want to try:
* opinion on adding self loop
* opinion on adding lstm, transformer in temporal block even if bad
* want to try one more spatial block, which one do you think is best ? --> gatconv from pygeo

Problems:
* static graph learning: you use node embedding, I don't, does it matter ?
* not enough data: How to prune in knn learn graph ? or do I redo preprocessing myself ?
* do I mention scalar STG I did at the beginning of the semester in the report ?
* scaling in chronos and CSDI: solved

Next:
* useful to add CRPS as metric ?
* Look for hyperparameters ? show tables in report
* Now I can rerun on bigger electricity benchamrk: how selective should I be on 
