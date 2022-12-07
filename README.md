# Modeling Event Plausibility with Consistent Conceptual Abstraction

## Data

### Physical Event Plausibility (PEP-3K)

Data from: https://github.com/suwangcompling/Modeling-Semantic-Plausibility-NAACL18
w/ additional pre-processing (spelling corrections and data splits)

### 20Q

Original data from https://github.com/allenai/twentyquestions w/ filtering and pre-processing as described in the paper

## src

Install requirements: ``pip install -r requirements.txt``
To finetune roberta run ``python src/train.py``
Model type can be specified by ``--model_type`` e.g. ``conceptmax``
