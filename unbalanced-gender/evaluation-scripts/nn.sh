#!/bin/zsh

DATASET_DIR=~/filteringNN/celebA/data/gender-gender/$1
NN_WEIGHTS_DIR=~/filteringNN/celebA/tests/gender-gender/$1/weight_nn
EXECUTABLE=~/filteringNN/celebA/src/gender-gender/reweight.py
OUTDIR=~/filteringNN/celebA/tests/gender-gender/$1/$2
mkdir $OUTDIR

for DATASET in $DATASET_DIR/*;
do
    DNAME=$(basename $DATASET)
    echo "NN $DNAME"
    # find the neural net file that corresponds to our dataset
    NN=$(find $NN_WEIGHTS_DIR  -maxdepth 1 -name "NN*$DNAME*.pt")
    if ls $OUTDIR/*$DNAME* 1> /dev/null 2>&1; then
        echo "$DNAME already exists, skipping..."
        continue
    else
        echo "$DNAME is missing!"
        CUDA_VISIBLE_DEVICES=$3 python3.6 $EXECUTABLE --data $DATASET --nn $NN --save-to $OUTDIR --epochs 120 --lr 0.02
    fi

done
