Natural Language Query Autocompletion for Segmentation
-----

Deep Learning Final Project 4995

Based largely on:
    
   - Personalized Language Model for Query Auto-Completion 
       * Paper: https://arxiv.org/pdf/1804.09661.pdf
       * github: https://github.com/ajaech/query_completion
       
   - Segmentation from Natural Language
Expressions 
       * Paper: https://arxiv.org/pdf/1603.06180.pdf
       * github: https://github.com/ronghanghu/text_objseg




### Currently Implemented

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
