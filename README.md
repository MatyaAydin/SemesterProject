# SemesterProject:

### Title:  Spatial-Temporal Denoising Diffusion Probabilistic Model for Industrial IoTs

Done with IMOS Lab at EPFL in collaboration with EWZ Zurich




### Dependencies

Create and activate virtual environment:
```bash

python3 -m venv venv
.\venv\Scripts\activate
```
Install packages
```
pip install -r requirements.txt
```

### Repo structure

* `src`: Code
* `data`: Datasets

#### Code structure

```
SemesterProject/
├── src/
   |
   ├── preprocessing/ # Scripts to create datasets
   │
   ├── diffstg/    # Diffusion-STG code
   │
   ├── CSDI/       # CSDI to benchmark
   │
   └── chronos/    # Chronos to benchmark

```

#### Reproducibility

CSDI results:

```bash

cd CSDI

python3 ./exe_forecasting.py --datatype "electricity_benchmark" --modelfolder "forecasting_electricity_benchmark" --nsample 8 # or ewz_daily

```

Chronos results:

```bash

cd chronos

python3 pred_chronos.py
```

DiffSTG results for all models:

```bash

cd diffstg

bash train_all.sh
```

To get a csv with all metrics:

```bash

cd diffstg/utils

python3 eval.py
```
