#!/bin/bash

set -o xtrace

DATA_DIR=~/perfect_shoot_data
DATASET=dataset_tiny
LABELS=dataset_labels.csv
RANKER_OUT=dataset_tiny

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

test_global() {
    python3 ranker.py --input $FEATURE_FILE --labels $LABELS_FILE --force \
        -svm \
        --global-image-quality $1\
        --color-image-quality ""\
        --content-image-quality ""\
        --face-features-quality ""\
        --output "ranker_out/${RANKER_OUT}/global_${1}"
}

test_color() {
    python3 ranker.py --input $FEATURE_FILE --labels $LABELS_FILE --force \
        -svm \
        --global-image-quality ""\
        --color-image-quality $1\
        --content-image-quality ""\
        --face-features-quality ""\
        --output "ranker_out/${RANKER_OUT}/color_${1}"
}

test_content() {
    python3 ranker.py --input $FEATURE_FILE --labels $LABELS_FILE --force \
        -svm \
        --global-image-quality ""\
        --color-image-quality ""\
        --content-image-quality $1\
        --face-features-quality ""\
        --output "ranker_out/${RANKER_OUT}/content_${1}"
}

test_faces() {
    python3 ranker.py --input $FEATURE_FILE --labels $LABELS_FILE --force \
        -svm \
        --global-image-quality ""\
        --color-image-quality ""\
        --content-image-quality ""\
        --face-features-quality $1\
        --output "ranker_out/${RANKER_OUT}/faces_${1}"
}

test_global_faces() {
    python3 ranker.py --input $FEATURE_FILE --labels $LABELS_FILE --force \
        -svm \
        --global-image-quality $1\
        --color-image-quality ""\
        --content-image-quality ""\
        --face-features-quality $2\
        --debug-images ~/perfect_shoot_data/dataset_feat\
        --output "ranker_out/${RANKER_OUT}/global_faces_${1}_${2}"
}

test_rfc_global_faces() {
    python3 ranker.py --input $FEATURE_FILE --labels $LABELS_FILE --force \
        -rfc \
        --global-image-quality $1\
        --color-image-quality ""\
        --content-image-quality ""\
        --face-features-quality $2\
        --output "ranker_out/${RANKER_OUT}/rfc_global_faces_${1}_${2}"
}

test_global_faces_color() {
    python3 ranker.py --input $FEATURE_FILE --labels $LABELS_FILE --force \
        -svm \
        --global-image-quality $1\
        --color-image-quality $3\
        --content-image-quality ""\
        --face-features-quality $2\
        --output "ranker_out/${RANKER_OUT}/global_faces_color_${1}_${2}_${3}"
}



# test_global "s"
# test_global "n"
# test_global "m"
# test_global "ms"
# test_global "mn"
# test_global "mns"

# test_color "c"
# test_color "s"
# test_color "cs"

# test_content "l"
# test_content "s"

# test_faces "c"
# test_faces "s"
# test_faces "e"
# test_faces "o"
# test_faces "b"

# test_faces "ec"
# test_faces "es"
# test_faces "eo"
# test_faces "eb"

# test_global_faces "m" "e"
# test_global_faces "m" "eo"
# test_global_faces "m" "es"
test_global_faces "m" "ec"
# test_global_faces "m" "eb"
# test_global_faces "m" "eco"

# test_global_faces "ms" "ec"
# test_global_faces "mn" "ec"

# test_global_faces_color "m" "ec" "s"
# test_global_faces_color "m" "ec" "c"

# test_rfc_global_faces "m" "ec"
