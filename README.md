# SemesterProject:

### Title:  Spatial-Temporal Denoising Diffusion Probabilistic Model for Industrial IoTs

Done with IMOS Lab at EPFL in collaboration with EWZ Zurich

#### Project stages:

1. Data Cleaning and Preprocessing
2. Graph Construction Based on Physical Poximity
3. Trying Some Baseline Graph Neural Networks (GNNs) with Physical Graph + Provide Comparison with other Graph Learning method
4. Pick the best graph (from stage 3) for Denoising Diffusion Probabilistic Model (DDPM)
5. Do Forecasting by Graph-Based DDPM and Anomaly Detection of Sensor Data
6. Possibility of Integrating Conservation Law


### Dependencies

Create a virtual environment:
```bash

python3 -m venv venv
```
Install packages
```
pip install -r requirements.txt
```


### Repo structure

* `papers`: Reference papers
* `notes`: Notes taken while reading papers/during meetings
* `src`: Code
* `data`: Datasets


### Other resources:

* [Article about spatio-temporal forecasting using GNNs](https://medium.com/data-reply-it-datatech/spatio-temporal-forecasting-using-temporal-graph-neural-networks-f27a8b326e5c)
* [Electricity dataset from Energy and AI paper](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014)
* [Pytorch geometric temporal notebook](https://colab.research.google.com/drive/132hNQ0voOtTVk3I4scbD3lgmPTQub0KR?usp=sharing)
* [Repo with links to paper about GNNs for fraud detection](https://github.com/safe-graph/graph-fraud-detection-papers)
* [DiffSTG repo](https://github.com/wenhaomin/DiffSTG)
* [TSL doc](https://torch-spatiotemporal.readthedocs.io/en/latest/)