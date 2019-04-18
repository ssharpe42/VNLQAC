import logging
import os
import time
import sys
import tensorflow as tf


# file_path = '/Users/Sam/Desktop/School/Deep Learning/FinalProject/NLQAC_ObjSeg/code/query_completion'
# os.chdir(file_path)
#sys.path.insert(0,'../')

from code.util import helper
from code.util.metrics import MovingAvg
from code.util.vgg.vgg_net import channel_mean
from code.query_completion.dataset import LoadData, Dataset
from code.query_completion.model import QACModel
from code.query_completion.vocab import Vocab

# Take out args for now

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
params = 'code/query_completion/default_params.json'
train_data = 'data/visual/train_image_queries.txt'
val_data = 'data/visual/val_image_queries.txt'
test_data = 'data/visual/test_image_queries.txt'
img_dir = 'data/visual/processed_images_224/'

# expdir = args.expdir
expdir = 'query_experiment'
if not os.path.exists(expdir):
    os.mkdir(expdir)
else:
    print 'ERROR: expdir already exists'
    exit(-1)

tf.set_random_seed(int(time.time() * 1000))

# params = helper.GetParams(args.params, 'train', args.expdir)
params = helper.GetParams(params, 'train', expdir)

logging.basicConfig(filename=os.path.join(expdir, 'logfile.txt'),
                    level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())

df = LoadData(train_data)
char_vocab = Vocab.MakeFromData(df.query_ , min_count=10)
char_vocab.Save(os.path.join(expdir, 'char_vocab.pickle'))
params.vocab_size = len(char_vocab)
dataset = Dataset(df, char_vocab,
                  max_len=params.max_len,
                  batch_size=params.batch_size,
                  image_dir=img_dir,
                  image_size=params.img_size,
                  only_char=only_char)

val_df = LoadData(val_data)
valdata = Dataset(val_df, char_vocab,
                  max_len=params.max_len,
                  batch_size=params.batch_size,
                  image_dir=img_dir,
                  image_size=params.img_size,
                  only_char=only_char)


test_df = LoadData(test_data)
testdata = Dataset(test_df, char_vocab,
                  max_len=params.max_len,
                  batch_size=params.batch_size,
                  image_dir=img_dir,
                  image_size=params.img_size,
                  only_char=only_char)


model = QACModel(params, only_char=only_char)
saver = tf.train.Saver(tf.global_variables())
config = tf.ConfigProto(inter_op_parallelism_threads=threads,
                        intra_op_parallelism_threads=threads,
                        allow_soft_placement=True)

# log_device_placement=True)

session = tf.Session(config=config)
session.run(tf.global_variables_initializer())

# Load pretrained vgg weights
model.LoadVGG(session, pretrained_path='code/util/vgg/vgg_params.npz')


avg_loss = MovingAvg(0.97)  # exponential moving average of the training loss
avg_val_loss = MovingAvg(0.9)  # exponential moving average of the val loss
for idx in range(params.iters):
    feed_dict = dataset.GetFeedDict(model, channel_mean=channel_mean)
    feed_dict[model.dropout_keep_prob] = params.dropout
    feed_dict[model.fc_dropout_keep_prob] = params.fc_dropout

    c, _ = session.run([model.avg_loss, model.train_op], feed_dict)
    cc = avg_loss.Update(c)
    if idx % 50 == 0:
        print('Iter: {}'.format(idx))
        # test one batch from the validation set
        val_c = session.run(model.avg_loss, valdata.GetFeedDict(model, channel_mean=channel_mean))
        vc = avg_val_loss.Update(val_c)
    if idx % 200 == 0 and idx > 0:
        logging.info({'iter': idx, 'cost': cc, 'rawcost': c, 'rawvalcost': val_c, 'valcost': vc})
    if idx % 2000 == 0:  # save a model file every 500 minibatches
        saver.save(session, os.path.join(expdir, 'model.bin'),write_meta_graph=False)

