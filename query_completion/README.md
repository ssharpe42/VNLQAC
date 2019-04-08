# Image Query Completion

This repo contains code for building an LSTM LM for query auto-completion with image as context.

Train a query model with ReferIt captions using 

python query_completion/code/trainer.py 
`
Set hyperparameters following the format in `default_params.json`.

Description of code files:
* beam.py - helper code for doing beam search
* factorcell.py - implementation of the FactorCell recurrent layer
* model.py - defines the Tensorflow graph for the language model
* trainer.py - script for training a new langauge model
* data/build_referit_data.py - preprocess referit queries to feed to LSTM
