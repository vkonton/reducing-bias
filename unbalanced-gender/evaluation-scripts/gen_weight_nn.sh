#!/bin/zsh

# $1 is the shuf dir 
# $2 is the name of the output dir

DATASET_DIR=~/filteringNN/celebA/data/gender-gender/$1
EXECUTABLE=~/filteringNN/celebA/src/gender-gender/gen_weight_nn.py
OUTDIR=~/filteringNN/celebA/tests/gender-gender/$1/$2
mkdir -p $OUTDIR

for DATASET in $DATASET_DIR/*;
do
    DNAME=$(basename $DATASET)
    if ls $OUTDIR/*$DNAME* 1> /dev/null 2>&1; then
        echo "$DNAME already exists, skipping..."
        continue
    else
        echo "$DNAME is missing!"
        CUDA_VISIBLE_DEVICES=$3 python3.6 $EXECUTABLE --data $DATASET --save-to $OUTDIR --epochs 100 --lr 0.1 --batch-size 500 --test-batch-size 800
    fi
done
