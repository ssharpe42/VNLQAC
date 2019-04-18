
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import json
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

#file_path = '/Users/Sam/Desktop/School/Deep Learning/FinalProject/NLQAC_ObjSeg/instance_selection/'
#os.chdir(file_path)
#pd.options.display.max_columns = 100

data = 'data/visual/query_classes.txt'
train_data = 'data/visual/train_query_classes.txt'
val_data = 'data/visual/val_query_classes.txt'
test_data = 'data/visual/test_query_classes.txt'
params = 'code/instance_selection/default_params.json'
expdir = 'selection_experiment'


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
from model import SelectionModel
from code.util import helper
from code.util.metrics import *
from bert_utils import create_tokenizer_from_hub_module


#params = helper.GetParams(args.params, 'train', args.expdir)
params = helper.GetParams(params, 'train', expdir)

# Load data
query_dict, class_indx, LABELS = LoadData(data, limit = None)

train = np.unique(pd.read_csv(train_data,sep = '\t',index_col=None)['query'])
val = np.unique(pd.read_csv(val_data,sep = '\t',index_col=None)['query'])
test = np.unique(pd.read_csv(test_data,sep = '\t',index_col=None)['query'])

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

test_dataset = Dataset(query_dict = query_dict,
                  query_set = test,
                  tokenizer = tokenizer,
                  num_labels = params.num_labels,
                  batch_size=32,
                  max_seq_len = params.max_seq_len )

model = SelectionModel(params)
saver = tf.train.Saver(tf.global_variables())
config = tf.ConfigProto(inter_op_parallelism_threads=2,
                        intra_op_parallelism_threads=2,
                        allow_soft_placement = True,
                        log_device_placement=True)

session = tf.Session(config=config)
session.run(tf.global_variables_initializer())



avg_loss = MovingAvg(0.97)  # exponential moving average of the training loss
avg_val_loss = MovingAvg(0.90)  # exponential moving average of the val loss
val_tp, val_fp, val_fn = [0]*3
for idx in range(params.iters):

    feed_dict = dataset.GetFeedDict(model)
    c, _ = session.run([model.avg_loss, model.train_op], feed_dict)
    cc = avg_loss.Update(c)

    if idx % 50 == 0:
        print('Iter: {}'.format(idx))
        # test one batch from the validation set
        val_c, tp, fp, fn = session.run([model.avg_loss,  model.tp, model.fp, model.fn], val_dataset.GetFeedDict(model))

        val_tp +=tp; val_fp +=fp; val_fn +=fn
        vc = avg_val_loss.Update(val_c)

    if idx % 1000 == 0 and idx > 0:
        val_f1 = f1(val_tp, val_fp, val_fn)

        logging.info({'iter': idx, 'cost': cc, 'rawcost': c, 'rawvalcost': val_c, 'valcost': vc,'valf1': val_f1})
        print({'iter': idx, 'cost': cc, 'rawcost': c, 'rawvalcost': val_c, 'valcost': vc,'valf1': val_f1})

        val_tp, val_fp, val_fn = [0]*3
    if idx % 5000 == 0:  # save a model file every 5000 minibatches
        saver.save(session, os.path.join(expdir, 'model.bin'),
                   write_meta_graph=False)


#Evaluate Test Set
logging.info('Evaluating Test Set...')

thresholds = np.linspace(.1,.6, num = 11)

for thresh in thresholds:
    test_loss = 0
    test_tp, test_fp, test_fn = [0]*3
    test_iters = len(test_dataset.query_set)//params.batch_size +1
    for idx in range(test_iters):

        test_c, tp, fp, fn = session.run([model.avg_loss, model.tp, model.fp, model.fn], val_dataset.GetFeedDict(model, prob_threshold=thresh))

        test_loss += test_c/test_iters; test_tp +=tp; test_fp+=fp; test_fn+=fn

    print({'test_loss':test_loss, 'test_f1': f1(test_tp, test_fp, test_fn),'threshold': thresh})
    logging.info({'test_loss':test_loss, 'test_f1': f1(tp, fp, fn),'threshold': thresh})