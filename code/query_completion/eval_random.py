import argparse
import logging
import numpy as np
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0],'..'))

from query_completion.dataset import LoadData, Dataset
from query_completion.model import  MetaQACModel
from util.vgg.vgg_net import channel_mean
from util.metrics import GetRankInList
from query_completion.beam import GetCompletions

parser = argparse.ArgumentParser()
parser.add_argument('expdir', help='experiment directory')
# parser.add_argument('--data', type=str, action='append', dest='data',
#                     help='where to load the data')
# parser.add_argument('--threads', type=int, default=12,
#                     help='how many threads to use in tensorflow')
args = parser.parse_args()
expdir = args.expdir

threads = 2
test_data = ['data/visual/test_image_queries.txt']
img_dir = {'visual':'data/visual/processed_images_224/'}


metamodel = MetaQACModel(expdir)
model = metamodel.model
metamodel.MakeSessionAndRestore(threads)

df = LoadData(test_data)
dataset = Dataset(df, metamodel.char_vocab,max_len=metamodel.params.max_len, image_dir=img_dir)

logging.basicConfig(filename=os.path.join(expdir, 'eval_results_random.txt'),
                    level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())
############ Perplexity with Images ##################
logging.info('---------- Perplexity -----------')

n_samples = len(dataset.df)
# total_word_count = 0
# total_log_prob = 0
# for idx in range(n_samples / dataset.batch_size):
#   feed_dict = dataset.GetFeedDict(model, channel_mean=channel_mean, random_img=False)
#   c, words_in_batch = metamodel.session.run([model.avg_loss, model.words_in_batch],feed_dict)
#
#   total_word_count += words_in_batch
#   total_log_prob += float(c * words_in_batch)
#
# logging.info('Perplexity: {:.5f}'.format(np.exp(total_log_prob / total_word_count)))
#
# ############ Perplexity with Random Image ##################
#
# total_word_count = 0
# total_log_prob = 0
# for idx in range(n_samples / dataset.batch_size):
#     feed_dict = dataset.GetFeedDict(model, channel_mean=channel_mean, random_img=True)
#     c, words_in_batch = metamodel.session.run([model.avg_loss, model.words_in_batch], feed_dict)
#
#     total_word_count += words_in_batch
#     total_log_prob += float(c * words_in_batch)
#
#
# logging.info('Perplexity (Random Images): {:.5f}'.format(np.exp(total_log_prob / total_word_count)))


################ Mean Reciprocal Rank #####################

logging.info('---------- Mean Reciprocal Rank -----------')

#
# query_pct = np.round( np.linspace(.1,.9,5),1)
# MRR_results = {q_pct:0 for q_pct in query_pct }
#
# sample_number = 1
# for image_id in np.unique(dataset.df.image_id):
#
#     queries = dataset.df[dataset.df.image_id==image_id]['query'].values
#     image = np.load(os.path.join(img_dir['visual'],str(image_id)+'.npy'))
#     metamodel.Lock(image-channel_mean)
#
#     for query in queries:
#
#         for q_pct in query_pct:
#
#             prefix = query[0:int(len(query)*q_pct)]
#             completions = list(GetCompletions(['<S>'] + list(prefix), metamodel, branching_factor=4, beam_size=50))[:-21:-1]
#             comp_list = [''.join(c.words) for c in  completions]
#
#             #Reciprocal Rank
#             RR = GetRankInList('<S>'+query+'</S>', comp_list)
#             MRR_results[q_pct] += 1.0*RR
#
#         if sample_number % 2000 ==0:
#             logging.info('Iteration {}'.format(sample_number))
#             logging.info({x:MRR_results[x]/(sample_number) for x in MRR_results})
#
#         sample_number+=1
#
#
# MRR_results = {x:MRR_results[x]/sample_number for x in MRR_results}
# logging.info('MRR:')
# logging.info(MRR_results)


### Random Image

query_pct = np.round( np.linspace(.1,.9,5),1)
MRR_results = {q_pct:0 for q_pct in query_pct }

sample_number = 1
image = np.random.random((224,224,3))
metamodel.Lock(image-channel_mean)

for query in np.unique(dataset.df['query']):

    for q_pct in query_pct:

        prefix = query[0:int(len(query)*q_pct)]
        completions = list(GetCompletions(['<S>'] + list(prefix), metamodel, branching_factor=4, beam_size=50))[:-11:-1]
        comp_list = [''.join(c.words) for c in  completions]

        #Reciprocal Rank
        RR = GetRankInList('<S>'+query+'</S>', comp_list)
        MRR_results[q_pct] += 1.0*RR

    if sample_number % 2000 == 0:
        logging.info('Iteration {}'.format(sample_number))
        logging.info({x: MRR_results[x] / (sample_number) for x in MRR_results})

    sample_number += 1


MRR_results = {x:MRR_results[x]/sample_number for x in MRR_results}
logging.info('MRR:')
logging.info(MRR_results)