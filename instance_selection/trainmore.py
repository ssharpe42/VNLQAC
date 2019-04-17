
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import json
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


file_path = '/Users/Sam/Desktop/School/Deep Learning/FinalProject/NLQAC_ObjSeg/instance_selection/'
os.chdir(file_path)
sys.path.insert(0,'code')
#pd.options.display.max_columns = 100


data = '../query_completion/data/visual/query_classes.txt'
params = 'code/default_params.json'
expdir = 'experiment1'


if not os.path.exists(expdir):
  os.mkdir(expdir)
else:
  print('ERROR: expdir already exists')
  exit(-1)


logging.basicConfig(filename=os.path.join(expdir, 'logfile.txt'),
                    level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())
logging.getLogger().setLevel(logging.INFO)


from dataset import LoadData, Dataset
from model import Model, MetaModel
import helper
from metrics import MovingAvg
from bert_utils import create_tokenizer_from_hub_module


#params = helper.GetParams(args.params, 'train', args.expdir)
params = helper.GetParams(params, 'train', expdir)

# Load data
query_dict, class_indx, LABELS = LoadData(data, limit = 1000)
train, val = train_test_split(query_dict.keys(), test_size=.2)

tokenizer = create_tokenizer_from_hub_module()
params.num_labels = len(query_dict[train[0]])

#Save in experiment
with open(os.path.join(expdir, 'params.json'),'w') as f:
    json.dump(params,f)
with open(os.path.join(expdir, 'class_indx.json'),'w') as f:
    json.dump(class_indx, f)

dataset = Dataset(query_dict = query_dict,
                  query_set = train,
                  tokenizer = tokenizer,
                  num_labels = params.num_labels,
                  batch_size=32,
                  max_seq_len = params.max_seq_len )

val_dataset = Dataset(query_dict = query_dict,
                  query_set = val,
                  tokenizer = tokenizer,
                  num_labels = params.num_labels,
                  batch_size=32,
                  max_seq_len = params.max_seq_len )

model = Model(params)
saver = tf.train.Saver(tf.global_variables())
config = tf.ConfigProto(inter_op_parallelism_threads=2,
                        intra_op_parallelism_threads=2,
                        allow_soft_placement = True,
                        log_device_placement=True)

session = tf.Session(config=config)
session.run(tf.global_variables_initializer())
saver.restore(session, os.path.join(expdir, 'model.bin'))


avg_loss = MovingAvg(0.97)  # exponential moving average of the training loss
avg_val_loss = MovingAvg(0.9)  # exponential moving average of the val loss
for idx in range(params.iters):

    feed_dict = dataset.GetFeedDict(model)
    c, _ = session.run([model.avg_loss, model.train_op], feed_dict)
    cc = avg_loss.Update(c)

    if idx % 2 == 0:
        print('Iter: {}'.format(idx))
        # test one batch from the validation set
        val_c, val_f1 = session.run([model.avg_loss,  model.f1], val_dataset.GetFeedDict(model))
        vc = avg_val_loss.Update(val_c)
    if idx % 2 == 0 and idx > 0:
        logging.info({'iter': idx, 'cost': cc, 'rawcost': c, 'rawvalcost': val_c, 'valcost': vc,'valf1': val_f1})
        print({'iter': idx, 'cost': cc, 'rawcost': c, 'rawvalcost': val_c, 'valcost': vc,'valf1': val_f1})
    if idx % 2000 == 0:  # save a model file every 500 minibatches
        saver.save(session, os.path.join(expdir, 'model.bin'),
                   write_meta_graph=False)


