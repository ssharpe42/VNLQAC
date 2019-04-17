from __future__ import absolute_import, division, print_function

import sys
import numpy as np
import pandas as pd
import os
import glob
import json
import re
import ijson
import skimage
import skimage.io
import skimage.transform
from sklearn.model_selection import train_test_split

#file_dir = os.path.dirname(os.path.abspath(__file__))
file_dir = '/Users/Sam/Desktop/School/Deep Learning/FinalProject/NLQAC_ObjSeg/query_completion/data'
os.chdir(file_dir)
util_location = os.path.normpath(os.path.join(file_dir, '../code/'))
sys.path.insert(0, util_location)
sys.path.insert(0, os.getcwd())

from util import im_processing, text_processing
from segment_everything_classes import segment_classes

################################################################################
# Parameters
################################################################################

image_dir = os.path.join('visual/images/')

# Saving directory
query_data_folder = 'visual'
processed_img_folder = 'visual/processed_images_224'
################################################################################
# Load annotations
################################################################################

images = glob.glob(image_dir+ '*.jpg')
image_ids = [int(re.search('[0-9]+',x).group()) for x in images]
f = open('visual/region_graphs.json','r')

VB_data = {'image_id':[],'query':[],'class':[],'x':[],'y': [],'h': [],'w': []}

for item in ijson.items(f, 'item'):
    img_id = item['image_id']
    if img_id % 100 ==0:
        print(img_id)
    for region in item['regions']:
        phrase = region['phrase']
        for obj in region['objects']:
            if len(obj['synsets'])>0:
                if img_id in image_ids:
                    VB_data['image_id'].append(img_id)
                else:
                    VB_data['image_id'].append('none')
                VB_data['query'].append(phrase)
                VB_data['class'].append(obj['synsets'][0])
                VB_data['x'].append(obj['x'])
                VB_data['y'].append(obj['y'])
                VB_data['h'].append(obj['h'])
                VB_data['w'].append(obj['w'])


VB_df = pd.DataFrame.from_dict(VB_data)
VB_image_query = VB_df[(~VB_df.duplicated(['image_id','query'])) & (VB_df['image_id']!='none')][['image_id','query']]

#Keep images with at least 40 queries
VB_image_query= VB_image_query.groupby('image_id').filter(lambda x: len(x)>=40)
sufficient_image_ids = np.unique(VB_image_query.image_id.values)

train_images, val_images = train_test_split(sufficient_image_ids, test_size=.15, random_state=42)
train_image_queries = VB_image_query[VB_image_query.image_id.isin(train_images)]
train_image_queries['dataset'] = 'visual'
val_image_queries = VB_image_query[VB_image_query.image_id.isin(val_images)]
val_image_queries['dataset'] = 'visual'

train_image_queries.to_csv(os.path.join(query_data_folder,'train_image_queries.txt'), sep='\t',index = False, encoding = 'utf-8')
val_image_queries.to_csv(os.path.join(query_data_folder,'val_image_queries.txt'),sep='\t', index = False, encoding = 'utf-8')


#Keep classes in Learning to Segment Everyting
VB_df['class'] = VB_df['class'].str.replace('\.[a-z0-9\.]+','')
VB_df = VB_df[VB_df['class'].isin(segment_classes)]
VB_query_classes = VB_df[~VB_df.duplicated(['query','class'])][['image_id','query','class']]
VB_class_instances = VB_df[~VB_df.duplicated(['image_id','class','x','y','w','h']) & (VB_df['image_id']!='none')]

#Write Query --> Class to csv
VB_query_classes.to_csv(os.path.join(query_data_folder,'query_classes.txt'), sep='\t', index = False, encoding = 'utf-8')
VB_class_instances.to_csv(os.path.join(query_data_folder,'image_class_instances.txt'), sep='\t', index = False, encoding = 'utf-8')



################################################################################
# Process images
################################################################################
if not os.path.isdir(processed_img_folder):
    os.mkdir(processed_img_folder)


# Model Params
input_H = 224
input_W = 224

images = np.unique(VB_image_query.image_id)
for i in range(len(images)):
    if i % 100 ==0:
        print('saving img {}: {}'.format(i,images[i]))

    imname = str(images[i])+'.jpg'
    im = skimage.io.imread(image_dir + imname)

    processed_im = skimage.img_as_ubyte(im_processing.resize_and_pad(im, input_H, input_W))
    if processed_im.ndim == 2:
        processed_im = processed_im[:, :, np.newaxis]

    np.save(file=os.path.join(processed_img_folder , imname[:-4] + '.npy'), arr=processed_im)
