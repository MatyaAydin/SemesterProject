#!/bin/bash

for config_file in params/electricity_benchmark/*.txt; do
    echo "=========================================="
    echo "Submitting job with config: $config_file"
    echo "=========================================="
    
    
    source "$config_file"
    
    # Export variables so sbatch can see them
    export DATA
    export GRAPH_METHOD
    export GC_TYPE
    export TEMPORAL_TYPE
    export K
    export T_H
    export N_SAMPLES
    
    
    sbatch ./train.run
    
    echo ""
done