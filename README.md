Natural Language Query Auto-completion for Instance Selection
-----

Deep Learning Final Project 4995

A novel addition to instance segmentation methods that selects instances from the image based on auto-completed user queries conditioned on image context.

#### Requirements

- Python 2.7 
- tensorflow==1.11.0

#### Downloading Data

To download Visual Genome Data:

`data/visual/download_visual_dataset.sh`

To download version of ReferIt data from Hu et al.:

`data/referit/download_referit_dataset.sh`

#### Downloading VGG
`code/util/vgg/download_vgg_params.sh`
#### Build Data

Run the following to set up data for training. 
```
python data/build_referit_data.py
python data/build_visual_data.py
```

#### Training 

###### Query Completion:

Set params: `code/query_completion/default_params.json`

Train: `python code/query_completion/trainer.py /path/to/new/experimentdir --data /path/to/training.txt --valdata /path/to/validation.txt`

###### Instance Selection:

Set params: `code/instance_selection/default_params.json`

Train: `python code/instance_selection/train.py /path/to/new/experimentdir 
--data /path/to/fulldataset.txt 
--traindata /path/to/training.txt
--valdata /path/to/validation.txt
--testdata /path/to/testing.txt`

#### Demo




Aspects of code adapted from:
    
   - Personalized Language Model for Query Auto-Completion 
       * Paper: https://arxiv.org/pdf/1804.09661.pdf
       * github: https://github.com/ajaech/query_completion
       
   - Segmentation from Natural Language
Expressions 
       * Paper: https://arxiv.org/pdf/1603.06180.pdf
       * github: https://github.com/ronghanghu/text_objseg

