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
file_dir = '/Users/Sam/Desktop/School/Deep Learning/FinalProject/NLQAC_ObjSeg/query_completion/data'
os.chdir(file_dir)
text_objseg_location = os.path.normpath(os.path.join(file_dir, '../../text_objseg/'))
sys.path.insert(0, text_objseg_location)

from util import im_processing, text_processing


################################################################################
# Parameters
################################################################################

image_dir = os.path.join(text_objseg_location,'exp-referit/referit-dataset/images/')
train_query_file = os.path.join(text_objseg_location,'exp-referit/data/referit_query_train.json')
val_query_file =os.path.join(text_objseg_location,'exp-referit/data/referit_query_val.json')
imsize_file = os.path.join(text_objseg_location,'exp-referit/data/referit_imsize.json')

# Saving directory
query_data_folder = 'referit'
img_folder = 'referit/processed_images_224'

# Image Size
input_H = 224
input_W = 224


################################################################################
# Load annotations
################################################################################

train_query_dict = json.load(open(train_query_file))
val_query_dict = json.load(open(val_query_file))
imsize_dict = json.load(open(imsize_file))
img_list = imsize_dict.keys()
train_imcrop_list = train_query_dict.keys()
val_imcrop_list = val_query_dict.keys()

################################################################################
# Collect training samples
################################################################################

train_images = []
train_queries = []
num_imcrop_train = len(train_imcrop_list)
for n_imcrop in range(num_imcrop_train):
    if n_imcrop % 200 == 0: print('processing %d / %d' % (n_imcrop + 1, num_imcrop_train))
    imcrop_name = train_imcrop_list[n_imcrop]

    # Image
    imname = imcrop_name.split('_', 1)[0]
    for q in train_query_dict[imcrop_name]:
        train_images.append(imname)
        train_queries.append(q)

val_images = []
val_queries = []
num_imcrop_val = len(val_imcrop_list)
for n_imcrop in range(num_imcrop_val):
    if n_imcrop % 200 == 0: print('processing %d / %d' % (n_imcrop + 1, num_imcrop_val))
    imcrop_name = val_imcrop_list[n_imcrop]

    # Image
    imname = imcrop_name.split('_', 1)[0]
    for q in val_query_dict[imcrop_name]:
        val_images.append(imname)
        val_queries.append(q)

################################################################################
# Save training samples to disk
################################################################################

if not os.path.isdir(query_data_folder):
    os.mkdir(query_data_folder)

train_df = pd.DataFrame({'query': train_queries, 'image_id': train_images})
train_df['dataset'] = 'referit'
train_df.to_csv(os.path.join(query_data_folder, 'train_queries.txt'), sep='\t', encoding='utf-8', index=False)

val_df = pd.DataFrame({'query': val_queries, 'image_id': val_images})
val_df['dataset'] = 'referit'
val_df.to_csv(os.path.join(query_data_folder, 'val_queries.txt'), sep='\t', encoding='utf-8', index=False)

################################################################################
# Process images
################################################################################
if not os.path.isdir(img_folder):
    os.mkdir(img_folder)


images = np.concatenate([np.unique(train_df.image_id), np.unique(val_df.image_id)])

for i in range(len(images)):

    if i%100==0:
        print('Image {}'.format(i))

    imname = images[i] +'.jpg'
    im = skimage.io.imread(image_dir + imname)

    processed_im = skimage.img_as_ubyte(im_processing.resize_and_pad(im, input_H, input_W))
    if processed_im.ndim == 2:
        processed_im = processed_im[:, :, np.newaxis]

    np.save(file=os.path.join(img_folder , imname[:-4] + '.npy'), arr=processed_im)
