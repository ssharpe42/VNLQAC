
import argparse
import logging
import os
import pandas as pd
import time
import datetime
import numpy as np
import tensorflow as tf

import sys
#file_path = os.path.dirname(__file__)
# file_path = '/Users/Sam/Desktop/School/Deep Learning/FinalProject/NLQAC_ObjSeg/query_completion/code_img'
#
# os.chdir('/Users/Sam/Desktop/School/Deep Learning/FinalProject/NLQAC_ObjSeg/query_completion')
# sys.path.insert(0,'code')
# pd.options.display.max_columns = 100

import helper
from dataset import Dataset, LoadData, LoadReferItData,ReferItDataset
from model import Model
from vgg_net import channel_mean
from metrics import MovingAvg
from vocab import Vocab


#Take out args for now

# parser = argparse.ArgumentParser()
# parser.add_argument('expdir', help='experiment directory')
# parser.add_argument('--params', type=str, default='default_params.json',
#                     help='json file with hyperparameters')
# parser.add_argument('--data', type=str, action='append', dest='data',
#                     help='where to load the data from')
# parser.add_argument('--valdata', type=str, action='append', dest='valdata',
#                     help='where to load validation data', default=[])
# parser.add_argument('--threads', type=int, default=12,
#                     help='how many threads to use in tensorflow')
# args = parser.parse_args()

threads = 2
params = 'code/default_params.json'
train_data = 'data/referit/train_queries.txt'
val_data = 'data/referit/val_queries.txt'
train_img_dir = 'data/referit/img_train/'
val_img_dir = 'data/referit/img_val/'

#expdir = args.expdir
expdir = 'referit_experiment_img'
if not os.path.exists(expdir):
  os.mkdir(expdir)
else:
  print 'ERROR: expdir already exists'
  exit(-1)


tf.set_random_seed(int(time.time() * 1000))

#params = helper.GetParams(args.params, 'train', args.expdir)
params = helper.GetParams(params, 'train', expdir)

logging.basicConfig(filename=os.path.join(expdir, 'logfile.txt'),
                    level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())

df = LoadReferItData(train_data)
char_vocab = Vocab.MakeFromData(df.query_, min_count=10)
char_vocab.Save(os.path.join(expdir, 'char_vocab.pickle'))
params.vocab_size = len(char_vocab)
dataset = ReferItDataset(df, char_vocab, max_len=params.max_len,
                        batch_size=params.batch_size,
                         image_dir = train_img_dir)

val_df = LoadReferItData(val_data)
valdata = ReferItDataset(val_df, char_vocab,  max_len=params.max_len,
                  batch_size=params.batch_size, image_dir = val_img_dir)

model = Model(params)
saver = tf.train.Saver(tf.global_variables())
config = tf.ConfigProto(inter_op_parallelism_threads=threads,
                        intra_op_parallelism_threads=threads,
                        allow_soft_placement = True,
                        log_device_placement=True)


session = tf.Session(config=config)
session.run(tf.global_variables_initializer())
saver.restore(session, os.path.join(expdir, 'model.bin'))

# avg_loss = MovingAvg(0.97)
# for idx in range(300000):
#   feed_dict = dataset.GetFeedDict(model)
#   feed_dict[model.dropout_keep_prob] = params.dropout
#   c, _ = session.run([model.avg_loss, model.train_op], feed_dict)
#   cc = avg_loss.Update(c)
#   if idx % 20 == 0 and idx > 0:
#     val_c = session.run(model.avg_loss, valdata.GetFeedDict(model))
#     logging.info({'iter': idx, 'cost': cc, 'rawcost': c,
#                   'valcost': val_c})
#   if idx % 999 == 0:
#     saver.save(session, os.path.join(expdir, 'model2.bin'),
#                write_meta_graph=False)

avg_loss = MovingAvg(0.97)  # exponential moving average of the training loss
avg_val_loss = MovingAvg(0.5)  # exponential moving average of the val loss
for idx in range(params.iters):
  feed_dict = dataset.GetFeedDict(model, channel_mean=channel_mean)
  feed_dict[model.dropout_keep_prob] = params.dropout

  c, _ = session.run([model.avg_loss, model.train_op], feed_dict)
  cc = avg_loss.Update(c)
  if idx % 50 ==0:

      print('Iter: {}'.format(idx))
      # test one batch from the validation set
      val_c = session.run(model.avg_loss, valdata.GetFeedDict(model, channel_mean=channel_mean))
      vc = avg_val_loss.Update(val_c)
  if idx % 200 == 0 and idx > 0:
    logging.info({'iter': idx, 'cost': cc, 'rawcost': c,'rawvalcost': val_c,'valcost': vc})
