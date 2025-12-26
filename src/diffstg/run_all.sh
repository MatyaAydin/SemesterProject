#!/bin/bash

for config_file in params/electricity_benchmark/*.txt; do  
    source "$config_file"
    
    export DATA
    export GRAPH_METHOD
    export GC_TYPE
    export TEMPORAL_TYPE
    export K
    export T_H
    export N_SAMPLES
      
    sbatch ./train.run
    
done