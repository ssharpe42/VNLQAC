from __future__ import absolute_import, division, print_function

import sys
import os
file_dir = '/Users/Sam/Desktop/School/Deep Learning/FinalProject/NLQAC_ObjSeg/'
os.chdir(file_dir)
sys.path.insert(0, 'data')



import numpy as np
import pandas as pd
import glob
import json
import re
import ijson

import skimage
import skimage.io
import skimage.transform
from sklearn.model_selection import train_test_split

import im_processing
from segment_everything_classes import segment_classes


################################################################################
# Parameters
################################################################################
#https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
image_dir = 'data/visual/images/'
region_file = 'data/visual/region_graphs.json'

# Saving directory
query_data_folder = 'data/visual'
processed_img_folder = 'data/visual/processed_images_224'
################################################################################
# Load annotations
################################################################################

images = glob.glob(image_dir+ '*.jpg')
image_ids = [int(re.search('[0-9]+',x).group()) for x in images]

f = open(region_file,'r')

VB_data = {'image_id':[], 'image_exists':[], 'query':[],'class':[],'x':[],'y': [],'h': [],'w': []}

for item in ijson.items(f, 'item'):
    img_id = item['image_id']
    if img_id % 100 ==0:
        print(img_id)
    for region in item['regions']:
        phrase = region['phrase']
        for obj in region['objects']:
            if len(obj['synsets'])>0:
                if img_id in image_ids:
                    VB_data['image_exists'].append(True)
                else:
                    VB_data['image_exists'].append(True)
                VB_data['image_id'].append(img_id)
                VB_data['query'].append(phrase)
                VB_data['class'].append(obj['synsets'][0])
                VB_data['x'].append(obj['x'])
                VB_data['y'].append(obj['y'])
                VB_data['h'].append(obj['h'])
                VB_data['w'].append(obj['w'])


# VB_df1 = pickle.load(open('data/visual/VB_df.pkl','r'))
# VB_df2 = pickle.load(open('data/visual/VB_df_noimage.pkl','r'))
# VB_df1['image_exists'] = True
# VB_df2['image_exists'] = False

VB_df = pd.DataFrame.from_dict(VB_data)

image_ids = np.unique(VB_df.image_id.values)
train_images, val_images = train_test_split(image_ids, test_size=.15, random_state=42)
val_images, test_images = train_test_split(val_images, test_size=.5, random_state=42)


VB_image_query = VB_df[(~VB_df.duplicated(['image_id','query'])) & (VB_df['image_exists'])][['image_id','query']]

#Keep images with at least 40 queries
VB_image_query = VB_image_query.groupby('image_id').filter(lambda x: len(x)>=40)
sufficient_image_ids = np.unique(VB_image_query.image_id.values)

query_train_images = np.intersect1d(train_images, sufficient_image_ids)
query_val_images = np.intersect1d(val_images, sufficient_image_ids)
query_test_images = np.intersect1d(test_images, sufficient_image_ids)

train_image_queries = VB_image_query[VB_image_query.image_id.isin(query_train_images)]
train_image_queries['dataset'] = 'visual'
val_image_queries = VB_image_query[VB_image_query.image_id.isin(query_val_images)]
val_image_queries['dataset'] = 'visual'
test_image_queries = VB_image_query[VB_image_query.image_id.isin(query_test_images)]
test_image_queries['dataset'] = 'visual'


train_image_queries.to_csv(os.path.join(query_data_folder,'train_image_queries.txt'), sep='\t',index = False, encoding = 'utf-8')
val_image_queries.to_csv(os.path.join(query_data_folder,'val_image_queries.txt'),sep='\t', index = False, encoding = 'utf-8')
test_image_queries.to_csv(os.path.join(query_data_folder,'test_image_queries.txt'),sep='\t', index = False, encoding = 'utf-8')


#Keep classes in Learning to Segment Everyting
VB_df['class'] = VB_df['class'].str.replace('\.[a-z0-9\.]+','')
VB_df = VB_df[VB_df['class'].isin(segment_classes)]
VB_query_classes = VB_df[~VB_df.duplicated(['query','class'])][['image_id','query','class']]
VB_class_instances = VB_df[~VB_df.duplicated(['image_id','class','x','y','w','h']) & (VB_df['image_exists'])]


train_query_classes = VB_query_classes[VB_query_classes.image_id.isin(train_images)]
val_query_classes = VB_query_classes[VB_query_classes.image_id.isin(val_images)]
test_query_classes = VB_query_classes[VB_query_classes.image_id.isin(test_images)]

#Write Query --> Class to csv
train_query_classes.to_csv(os.path.join(query_data_folder,'train_query_classes.txt'), sep='\t',index = False, encoding = 'utf-8')
val_query_classes.to_csv(os.path.join(query_data_folder,'val_query_classes.txt'),sep='\t', index = False, encoding = 'utf-8')
test_query_classes.to_csv(os.path.join(query_data_folder,'test_query_classes.txt'),sep='\t', index = False, encoding = 'utf-8')


#All instances tied to images (not currently used)
VB_class_instances.to_csv(os.path.join(query_data_folder,'image_class_instances.txt'), sep='\t', index = False, encoding = 'utf-8')

################################################################################
# Process images
################################################################################
if not os.path.isdir(processed_img_folder):
    os.mkdir(processed_img_folder)


# Image Size
input_H = 224
input_W = 224

images = np.unique(np.concatenate([train_image_queries.image_id.values,
                                   val_image_queries.image_id.values,
                                   test_image_queries.image_id.values]))
for i in range(len(images)):
    if i % 100 ==0:
        print('saving img {}: {}'.format(i,images[i]))

    imname = str(images[i])+'.jpg'
    im = skimage.io.imread(image_dir + imname)

    processed_im = skimage.img_as_ubyte(im_processing.resize_and_pad(im, input_H, input_W))
    if processed_im.ndim == 2:
        processed_im = processed_im[:, :, np.newaxis]

    np.save(file=os.path.join(processed_img_folder , imname[:-4] + '.npy'), arr=processed_im)


