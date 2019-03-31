from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
import os
import json
import skimage
import skimage.io
import skimage.transform

from util import im_processing, text_processing

################################################################################
# Parameters
################################################################################

image_dir = './exp-referit/referit-dataset/images/'
mask_dir = './exp-referit/referit-dataset/mask/'
train_query_file = './exp-referit/data/referit_query_train.json'
val_query_file = './exp-referit/data/referit_query_val.json'
imsize_file = './exp-referit/data/referit_imsize.json'
vocab_file = './exp-referit/data/vocabulary_referit.txt'

# Saving directory
query_data_folder = './exp-referit/data/referit_query_data/'

# Model Params
T = 20
N = 100
input_H = 512; featmap_H = (input_H // 32)
input_W = 512; featmap_W = (input_W // 32)

################################################################################
# Load annotations
################################################################################

train_query_dict = json.load(open(train_query_file))
val_query_dict = json.load(open(val_query_file))
imsize_dict = json.load(open(imsize_file))
img_list = imsize_dict.keys()
train_imcrop_list = train_query_dict.keys()
val_imcrop_list = val_query_dict.keys()
vocab_dict = text_processing.load_vocab_dict_from_file(vocab_file)

################################################################################
# Collect training samples
################################################################################

train_images = []
train_queries = []
num_imcrop_train = len(train_imcrop_list)
for n_imcrop in range(num_imcrop_train):
    if n_imcrop % 200 == 0: print('processing %d / %d' % (n_imcrop+1, num_imcrop_train))
    imcrop_name = train_imcrop_list[n_imcrop]

    # Image
    imname = imcrop_name.split('_', 1)[0] + '.jpg'
    for q in train_query_dict[imcrop_name]:
        train_images.append(imname)
        train_queries.append(q)
        
val_images = []
val_queries = []
num_imcrop_val = len(val_imcrop_list)
for n_imcrop in range(num_imcrop_val):
    if n_imcrop % 200 == 0: print('processing %d / %d' % (n_imcrop+1, num_imcrop_val))
    imcrop_name = val_imcrop_list[n_imcrop]

    # Image
    imname = imcrop_name.split('_', 1)[0] + '.jpg'
    for q in val_query_dict[imcrop_name]:
        val_images.append(imname)
        val_queries.append(q)

################################################################################
# Save training samples to disk
################################################################################

if not os.path.isdir(query_data_folder):
    os.mkdir(query_data_folder)


train_df = pd.DataFrame({'queries':train_queries, 'images':train_images})
train_df.to_csv(os.path.join(query_data_folder, 'train_queries.txt'), sep='\t', encoding = 'utf-8', index = False)

val_df = pd.DataFrame({'queries':val_queries, 'images':val_images})
val_df.to_csv(os.path.join(query_data_folder, 'val_queries.txt'), sep='\t', encoding = 'utf-8', index = False)

#
# images = np.unique(images)
# for i in range(len(images)):
#     print('saving img {}: {}'.format(i,images[i]))
#     imname, description = queries.iloc[i]
#     im = skimage.io.imread(image_dir + imname)
#
#     processed_im = skimage.img_as_ubyte(im_processing.resize_and_pad(im, input_H, input_W))
#     if processed_im.ndim == 2:
#         processed_im = processed_im[:, :, np.newaxis]
#
#     np.save(file=query_data_folder + imname[:-4] + '.npy', arr=processed_im)
