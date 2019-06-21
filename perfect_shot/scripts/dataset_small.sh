#!/bin/bash

# output bash commands to console
set -o xtrace

# stop on first error
set -e

DATA_DIR=~/perfect_shoot_data
DATASET=dataset_small
LABELS=dataset_labels.csv
RANKER_OUT=dataset_small

FEATURIZE_PY=${PWD}/feature_extraction.py
FEATURE_FILE="${PWD}/dataset/${DATASET}_feat.csv"
LABELS_FILE="${PWD}/dataset/${DATASET}_labels.csv"

if [ ! -d "$DATA"]; then
    echo "Fatal: $DATA_DIR not found"
fi

if [ ! -d "$DATA_DIR/$DATASET" ]; then
    echo "Fatal: $DATA_DIR/$DATASET not found"
fi

if [ ! -f "$DATA_DIR/$LABELS" ]; then
    echo "Fatal: $DATA_DIR/$LABELS not found"
fi

# COPY LABELS, WE CONSIDER THAT DATADIR HAS CORRECT LABELS
cp $DATA_DIR/$LABELS $LABELS_FILE

# FEATURIZATION
pushd $DATA_DIR
python3 $FEATURIZE_PY -ip $DATASET -do ${DATASET}_feat -o $FEATURE_FILE --force
popd

python3 ranker.py --input $FEATURE_FILE --labels $LABELS_FILE --output "ranker_out/${RANKER_OUT}" -di $DATA_DIR/${DATASET}_feat  --force
