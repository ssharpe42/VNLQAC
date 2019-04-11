
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
from dataset import  LoadVisualData, VisualDataset
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
only_char = False
params = 'code/default_params.json'
train_data = 'data/visual/train_image_queries.txt'
val_data = 'data/visual/train_image_queries.txt'
img_dir = 'data/visual/processed_images_224/'


#expdir = args.expdir
expdir = 'visual_experiment'
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

df = LoadVisualData(train_data)
char_vocab = Vocab.MakeFromData(df.query_, min_count=10)
char_vocab.Save(os.path.join(expdir, 'char_vocab.pickle'))
params.vocab_size = len(char_vocab)
dataset = VisualDataset(df, char_vocab, max_len=params.max_len,
                        batch_size=params.batch_size,
                         image_dir = img_dir,
                        image_size=params.img_size,
                         only_char=only_char)

val_df = LoadVisualData(val_data)
valdata = VisualDataset(val_df, char_vocab,  max_len=params.max_len,
                  batch_size=params.batch_size, image_dir = img_dir,
                        image_size = params.img_size,
                         only_char=only_char)

model = Model(params,only_char=only_char)
saver = tf.train.Saver(tf.global_variables())
config = tf.ConfigProto(inter_op_parallelism_threads=threads,
                        intra_op_parallelism_threads=threads,
                        allow_soft_placement = True)#,
                        #log_device_placement=True)

session = tf.Session(config=config)
session.run(tf.global_variables_initializer())

#Load pretrained vgg weights
model.LoadVGG(session, pretrained_path='data/weights/vgg_params.npz')

session.run([var for var in tf.trainable_variables() if 'fc7/w' in var.name][0])[0]
#0.00390148, -0.00180807,  0.00136159

# pretrained_vgg = 'data/weights/vgg_params.npz'
# vgg_weights = np.load( pretrained_vgg)
#
# vgg_W = vgg_weights['processed_W'].item()
# vgg_B = vgg_weights['processed_B'].item()
#
# for name in vgg_W:
#     if name != 'fc8':
#         print(name)
#         weight = [w for w in tf.trainable_variables() if name in w.name and 'weight' in w.name][0]
#         print(weight)
#         session.run(weight.assign(vgg_W[name]))
#
# for name in vgg_B:
#     if name != 'fc8':
#         print(name)
#         weight = [w for w in tf.trainable_variables() if name in w.name and 'biases' in w.name][0]
#         print(weight)
#         session.run(weight.assign(vgg_B[name]))


avg_loss = MovingAvg(0.97)  # exponential moving average of the training loss
avg_val_loss = MovingAvg(0.5)  # exponential moving average of the val loss
for idx in range(params.iters):
  feed_dict = dataset.GetFeedDict(model, channel_mean=channel_mean)
  feed_dict[model.dropout_keep_prob] = params.dropout
  feed_dict[model.fc_dropout_keep_prob] = params.fc_dropout

  c, _ = session.run([model.avg_loss, model.train_op], feed_dict)
  cc = avg_loss.Update(c)
  if idx % 1 ==0:
      print('Iter: {}'.format(idx))
      # test one batch from the validation set
      val_c = session.run(model.avg_loss, valdata.GetFeedDict(model, channel_mean=channel_mean))
      vc = avg_val_loss.Update(val_c)
  if idx % 200 == 0 and idx > 0:
    logging.info({'iter': idx, 'cost': cc, 'rawcost': c,'rawvalcost': val_c,'valcost': vc})
  if idx % 2000 == 0:  # save a model file every 500 minibatches
    saver.save(session, os.path.join(expdir, 'model.bin'),
               write_meta_graph=False)



with session:
    writer = tf.summary.FileWriter("output", session.graph)
    print(session.run(model.per_word_loss))
    writer.close()