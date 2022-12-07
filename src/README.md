
### setting up environment

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

### syncing data

rsync -v -r -a --delete ~/Documents/current_projects/plausibility_rewrite/src/ beluga:~/plausibility_rewrite/src/

rsync -v -r -a --delete ~/Documents/current_projects/plausibility_rewrite/data/ beluga:~/plausibility_rewrite/data/

rsync --progress -v -r -a --delete ~/cache/models/ beluga:~/scratch/cache/models/

### running train

<!-- TRANSFORMERS_CACHE=$SCRATCH/cache/transformers/
mkdir -p $TRANSFORMERS_CACHE -->
<!-- rm -rf $CACHE_DIR/* -->

MODEL_PATH=$SCRATCH/cache/models/roberta-base
DATA_DIR=$SCRATCH/plausibility_rewrite/data
OUTPUT_DIR=$SCRATCH/output/
CACHE_DIR=$SCRATCH/cache/data/

mkdir $OUTPUT_DIR
mkdir $CACHE_DIR



cd ~/plausibility_rewrite/
python src/train.py \
    --model_name_or_path $MODEL_PATH \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --cache_dir $CACHE_DIR \
    --model_type conceptmax
