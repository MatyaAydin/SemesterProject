# Guidelines

## Preprocessing

* Keep only time series that make sense (i.e, roughly periodic over the year and max values during cold months)
* Once done, take intersection of dates where all sensors have values

## Graph construction:

* Build adjacency matrix based on distance. either raw $\Vert \cdot \Vert_2$ or with gaussian kernel then apply threshold or [Learn graph representation](https://github.com/andreacini/corel/blob/main/lib/nn/layers/knn_graph_learning.py).

## Metrics:

For first benchmark with scalar output: MAE, RMSE MAE percentage