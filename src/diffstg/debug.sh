#!/bin/bash

echo "Current directory: $(pwd)"
echo "Config files found:"
ls -la params/ewz_daily/*.txt

for config_file in params/ewz_daily/*.txt; do
    echo "=========================================="
    echo "Processing: $config_file"
    
    # Check if file exists and is readable
    if [ ! -f "$config_file" ]; then
        echo "ERROR: File not found: $config_file"
        continue
    fi
    
    echo "File contents:"
    cat "$config_file"
    echo "---"
    
    # Clear variables
    unset DATA GRAPH_METHOD GC_TYPE TEMPORAL_TYPE K T_H N_SAMPLES
    
    # Source the config
    source "$config_file"
    
    # Debug output
    echo "After sourcing:"
    echo "  T_H='$T_H'"
    echo "  K='$K'"
    echo "  N_SAMPLES='$N_SAMPLES'"
    
    if [ -z "$T_H" ]; then
        echo "ERROR: T_H is still empty after sourcing!"
        continue
    fi
    
done