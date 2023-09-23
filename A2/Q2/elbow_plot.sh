#!/bin/bash

dataset="$1"
dimension="$2"
plot_name="$3"

python q2_kmeans.py "$dataset" "$dimension" "$plot_name"
