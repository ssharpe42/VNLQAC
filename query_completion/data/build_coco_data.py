from __future__ import absolute_import, division, print_function

import sys
import numpy as np
import pandas as pd
import os
import json
import skimage
import skimage.io
import skimage.transform

#file_dir = os.path.dirname(os.path.abspath(__file__))
file_dir = '/Users/Sam/Desktop/School/Deep Learning/FinalProject/query_completion/data'
text_objseg_location = os.path.normpath(os.path.join(file_dir, '../../text_objseg/'))
sys.path.insert(0, text_objseg_location)

from util import im_processing


################################################################################
# Parameters
################################################################################

image_dir = 'data/coco/images'
train_query_file = 'data/coco/captions_train2017.json'
val_query_file ='data/coco/captions_val2017.json'

# Saving directory
query_data_folder = 'data/coco/'

# Model Params
T = 20
N = 100
input_H = 512;
featmap_H = (input_H // 32)
input_W = 512;
featmap_W = (input_W // 32)

################################################################################
# Load annotations
################################################################################

train_query_dict = json.load(open(train_query_file))
val_query_dict = json.load(open(val_query_file))

################################################################################
# Collect training samples
################################################################################

train_image_files = []
train_image_ids = []
train_queries = []
train_query_ids = []
train_ann_list = train_query_dict['annotations']
train_image_list = train_query_dict['images']

for n in range(len(train_ann_list)):
    train_queries.append(train_ann_list[n]['caption'])
    train_query_ids.append(train_ann_list[n]['image_id'])

for n in range(len(train_image_list)):
    train_image_files.append(train_image_list[n]['file_name'])
    train_image_ids.append(train_image_list[n]['id'])


val_image_files = []
val_image_ids = []
val_queries = []
val_query_ids = []
val_ann_list = val_query_dict['annotations']
val_image_list = val_query_dict['images']

for n in range(len(val_ann_list)):
    val_queries.append(val_ann_list[n]['caption'])
    val_query_ids.append(val_ann_list[n]['image_id'])

for n in range(len(val_image_list)):
    val_image_files.append(val_image_list[n]['file_name'])
    val_image_ids.append(val_image_list[n]['id'])

################################################################################
# Save training samples to disk
################################################################################

if not os.path.isdir(query_data_folder):
    os.mkdir(query_data_folder)

query_train_df = pd.DataFrame({'queries': train_queries, 'image_id': train_query_ids})
image_train_df = pd.DataFrame({'images': train_image_files, 'image_id': train_image_ids})
train_df = query_train_df.merge(image_train_df)
train_df.to_csv(os.path.join(query_data_folder, 'train_queries.txt'), sep='\t', encoding='utf-8', index=False)


query_val_df = pd.DataFrame({'queries': val_queries, 'image_id': val_query_ids})
image_val_df = pd.DataFrame({'images': val_image_files, 'image_id': val_image_ids})
val_df = query_val_df.merge(image_val_df)
val_df.to_csv(os.path.join(query_data_folder, 'val_queries.txt'), sep='\t', encoding='utf-8', index=False)

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
