import hashlib
import json
import os
import sys

import bunch

sys.path.insert(0, '../../data')

import numpy as np
import skimage.io
import skimage.transform
from data import im_processing


def GetPrefixLen(user, query, n=None):
    # choose a random prefix length
    hasher = hashlib.md5()
    hasher.update(user)
    hasher.update(''.join(query))
    if n:
        hasher.update(str(n))
    prefix_len = int(hasher.hexdigest(), 16) % (len(query) - 1)
    prefix_len += 1  # always have at least a single character prefix
    return prefix_len


def GetParams(filename, mode, expdir):
    param_filename = os.path.join(expdir, 'params.json')
    if mode == 'train':
        with open(filename, 'r') as f:
            param_dict = json.load(f)
            params = bunch.Bunch(param_dict)
        with open(param_filename, 'w') as f:
            json.dump(param_dict, f)
    else:
        with open(param_filename, 'r') as f:
            params = bunch.Bunch(json.load(f))
    return params


def TransformImage(filename, H, W):
    im = skimage.io.imread(filename)

    processed_im = skimage.img_as_ubyte(im_processing.resize_and_pad(im, H, W))
    if processed_im.ndim == 2:
        processed_im = processed_im[:, :, np.newaxis]

    return processed_im