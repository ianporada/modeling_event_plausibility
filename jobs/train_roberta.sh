#!/bin/bash
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=45000MB
#SBATCH --time=24:00:00

module purge

module load StdEnv/2020  gcc/9.3.0  cuda/11.7
module load arrow python/3.9 scipy-stack

ENVDIR=$SLURM_TMPDIR/env
virtualenv --no-download $ENVDIR
source $ENVDIR/bin/activate

pip install --no-index --upgrade pip

pip install --no-index torch
pip install --no-index scikit_learn

pip install --no-index datasets
pip install --no-index transformers
pip install --no-index pytorch-lightning

pip install --no-index click

REQUIREMENTS_DIR=$SCRATCH/logging/requirements/$SLURM_JOB_ID
mkdir -p $REQUIREMENTS_DIR
pip freeze > $REQUIREMENTS_DIR/requirements.txt

DATA_DIR=$SCRATCH/modeling_event_plausibility/data
OUTPUT_DIR=$SCRATCH/output/
CACHE_DIR=$SCRATCH/cache/data/

mkdir $OUTPUT_DIR
mkdir $CACHE_DIR

cd ~/modeling_event_plausibility/

python src/train.py \
    --model_name_or_path $MODEL_PATH \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --cache_dir $CACHE_DIR \
    --model_type roberta
