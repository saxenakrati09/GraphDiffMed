#!/bin/bash

# Script to run relevance construction for different datasets
# Usage: ./run_relevance_construction.sh [dataset]
# Default dataset is mimic3

DATASET=${1:-mimic3}

echo "Running relevance construction for dataset: $DATASET"

python Relevance_construction.py --dataset "$DATASET"

echo "Relevance construction completed for $DATASET"