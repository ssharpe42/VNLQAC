
import argparse
import logging
import os
import pandas as pd
import time
import datetime
import numpy as np
import tensorflow as tf

import sys
sys.path.insert(0,'code')
pd.options.display.max_columns = 100

import helper
from dataset import Dataset, LoadData, LoadReferItData
from model import Model
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
# train_data = 'data/coco/train_queries.txt'
# val_data = 'data/coco/val_queries.txt'


#expdir = args.expdir
expdir = 'referit_experiment1'
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
user_vocab = Vocab.MakeFromData([[u] for u in df.user], min_count=15)
user_vocab.Save(os.path.join(expdir, 'user_vocab.pickle'))
params.user_vocab_size = len(user_vocab)
dataset = Dataset(df, char_vocab, user_vocab, max_len=params.max_len,
                  batch_size=params.batch_size)

val_df = LoadReferItData(val_data)
valdata = Dataset(val_df, char_vocab, user_vocab, max_len=params.max_len,
                  batch_size=params.batch_size)

model = Model(params)
saver = tf.train.Saver(tf.global_variables())
config = tf.ConfigProto(inter_op_parallelism_threads=threads,
                        intra_op_parallelism_threads=threads,
                        log_device_placement=True)
session = tf.Session(config=config)
session.run(tf.global_variables_initializer())


avg_loss = MovingAvg(0.97)  # exponential moving average of the training loss
for idx in range(params.iters):
  feed_dict = dataset.GetFeedDict(model)
  feed_dict[model.dropout_keep_prob] = params.dropout

  c, _ = session.run([model.avg_loss, model.train_op], feed_dict)
  cc = avg_loss.Update(c)
  if idx % 50 == 0 and idx > 0:
    # test one batch from the validation set
    val_c = session.run(model.avg_loss, valdata.GetFeedDict(model))
    logging.info({'iter': idx, 'cost': cc, 'rawcost': c,
                  'valcost': val_c})
  if idx % 500 == 0:  # save a model file every 2,000 minibatches
    saver.save(session, os.path.join(expdir, 'model.bin'),
               write_meta_graph=False)




with session:
    writer = tf.summary.FileWriter("output", session.graph)
    print(session.run(model.per_word_loss))
    writer.close()