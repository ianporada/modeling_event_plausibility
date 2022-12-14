# Modeling Event Plausibility with Consistent Conceptual Abstraction

## Data

Download `data.zip` from [this Google Drive directory](https://drive.google.com/drive/folders/1KTKjVtb4xZ5TqFFQJq18OQl0_Pmso_qC?usp=sharing) and place the `./data` folder in the project directory. You can also place the data in a different location which must be referenced using the `--data_dir` argument.

This directory contains:
1. Wikipedia training data as a parquet (pyarrow) table. Each row is a training example with a uttered s-v-o triple and it's corresponding pseudo-negative. Specifcally, the columns are ```(subject verb object negative_subject negative_verb negative_object subject_synset object_synset negative_subject_synset negative_object_synset)``` where synsets are those disambiguated using BERT-WSD.
1. Plausibility judgements for evaluation (PEP-3K and Twenty Questions) in `.tsv` format.
1. Filtered WordNet saved a `.tsv` files. `lemma2synsets` is a mapping from lemmas to corresponding synsets. `synset2hc` is a mapping from synset to hypernym chain. `synset2lemma` is a mapping from synset to lemma.

## Requirements

First make sure you have the requirements as specified in the `requirements.txt` file.
E.g., create a new virtual environment and install the necessary requirements:
```bash
virtualenv ./env
source ./env/bin/activate
pip install -r requirements.txt
```

## Training the model

To train a model, and run (specifying the model type, i.e. `roberta` or `conceptmax`):
```bash
python src/train.py \
    --model_type roberta
```

The training dataset and WordNet data will be cached the first time training is run.

You can override the default directories and also resume training from an existing checkpoint:
```bash
python src/train.py \
    --model_name_or_path $MODEL_PATH \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --cache_dir $CACHE_DIR \
    --model_type conceptmax \
    --ckpt_path $CHECKPOINT_DIR/last.ckpt \
    --stage train
```

I haven't merged in `conceptinject` as the performance is similar to `roberta`. Please let me know if you need this model.

### Testing the model

You can test a model by specifying `test` as the `--stage` and pointing to the pytorch-lightning checkpoint to be evaluated, e.g.:
```bash
python src/train.py \
    --model_type conceptmax \
    --ckpt_path $CHECKPOINT_DIR/last.ckpt \
    --stage test
```

#### Trained checkpoints

The [Google Drive directory](https://drive.google.com/drive/folders/1KTKjVtb4xZ5TqFFQJq18OQl0_Pmso_qC?usp=sharing) also has pytorch-checkpoints for trained models. You can, for example, evaluate these models by downloading the relevant `.ckpt` file and then running the test stage:
```bash
python src/train.py \
    --model_type conceptmax \
    --ckpt_path ./roberta-plausibility.ckpt \
    --stage test
```

The AUC results of these models are higher than those reported in the paper. I think this might be due to a smaller Wikipedia validation split (and thus larger training set):

<table>
  <tr>
    <td rowspan="2">Model</td>
    <td colspan="2">PEP-3K</td>
    <td colspan="2">20 Questions</td>
  </tr>
  <tr>
    <td>Valid</td>
    <td>Test</td>
    <td>Valid<t/d>
    <td>Test</td>
  </tr>
  <tr>
    <td>Roberta</td>
    <td>0.702</td>
    <td>0.678</td>
    <td>0.692<t/d>
    <td>0.688</td>
  </tr>
  <tr>
    <td>ConceptMax</td>
    <td>0.679</td>
    <td>0.698</td>
    <td>0.746<t/d>
    <td>0.757</td>
  </tr>
</table>

### Running with Slurm

Models can be run using Slurm Workload Manager. See [./jobs](jobs/README.md)
