# Guidelines

## Preprocessing

* Keep only time series that make sense (i.e, roughly periodic over the year and max values during cold months)
* Once done, take intersection of dates where all sensors have values
--> best results by manual inspection

## Graph construction

No spatial data -->[Learn graph representation](https://github.com/andreacini/corel/blob/main/lib/nn/layers/knn_graph_learning.py).

ablation study: comapre with vs without graph learning, and other blocks and eletrical dataset

anomaly detection: build many c% CI and classify anomalies

change block in ugnet one by one then compare
IOT: electricty data: open dataset and no need to learn graph (just to benchmark) use get_connectivity from tsl and compare with knn lernable

## Metrics:

For first benchmark with scalar output: MAE, RMSE MAE percentage

**TODO**:

* Extract and visualize learned adjacency matrix
* Clean repo
