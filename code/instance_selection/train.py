
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import sys
import argparse
import json
import time
import pandas as pd
import tensorflow as tf

sys.path.insert(1, os.path.join(sys.path[0],'..'))


parser = argparse.ArgumentParser()
parser.add_argument('expdir', help='experiment directory')
parser.add_argument('--params', type=str, default='code/instance_selection/default_params.json',
                    help='json file with hyperparameters')
parser.add_argument('--data', type=str,  dest='data', default= 'data/visual/query_classes.txt',
                    help='where to load the data')
parser.add_argument('--traindata', type=str,  dest='traindata', default= 'data/visual/train_query_classes.txt',
                    help='where to load the training data')
parser.add_argument('--valdata', type=str,  dest='valdata', default= 'data/visual/val_query_classes.txt',
                    help='where to load the validation data')
parser.add_argument('--testdata', type=str,  dest='testdata', default= 'data/visual/test_query_classes.txt',
                    help='where to load the test data')
parser.add_argument('--threads', type=int, default=2,
                    help='how many threads to use in tensorflow')
args = parser.parse_args()


data = args.data
train_data = args.traindata
val_data = args.valdata
test_data = args.testdata
params = args.params
expdir = args.expdir


if not os.path.exists(expdir):
  os.mkdir(expdir)
else:
  print('ERROR: expdir already exists')
  exit(-1)

tf.set_random_seed(int(time.time() * 1000))
logging.basicConfig(filename=os.path.join(expdir, 'logfile.txt'),
                    level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())
logging.getLogger().setLevel(logging.INFO)


from instance_selection.dataset import LoadData, Dataset
from instance_selection.model import SelectionModel
from util import helper
from util.metrics import *
from instance_selection.bert_utils import create_tokenizer_from_hub_module


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
config = tf.ConfigProto(inter_op_parallelism_threads=args.threads,
                        intra_op_parallelism_threads=args.threads,
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
        val_dict = val_dataset.GetFeedDict(model)
        val_dict[model.dropout] = 1.0
        val_c, tp, fp, fn = session.run([model.avg_loss,  model.tp, model.fp, model.fn], val_dict)

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

        test_dict = test_dataset.GetFeedDict(model, prob_threshold=thresh)
        test_dict[model.dropout] = 1.0
        test_c, tp, fp, fn = session.run([model.avg_loss, model.tp, model.fp, model.fn], test_dict)

        test_loss += test_c/test_iters; test_tp +=tp; test_fp+=fp; test_fn+=fn

    print({'test_loss':test_loss, 'test_f1': f1(test_tp, test_fp, test_fn),'threshold': thresh})
    logging.info({'test_loss':test_loss, 'test_f1': f1(tp, fp, fn),'threshold': thresh})