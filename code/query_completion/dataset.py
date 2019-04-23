"""Holds the Dataset class used for managing training and test data."""
import datetime
import os
import numpy as np
import pandas as pd


#
# def LoadData(filename, split=True):
#     """Load a bunch of files as a pandas dataframe.
#
#     """
#     def Prepare(s):
#         s = str(s)
#         return ['<S>'] + list(s) + ['</S>']
#
#     df = pd.read_csv(filename, sep='\t')
#
#     if split:
#         df['query_'] = df['query'].apply(Prepare)
#
#     return df



def LoadData(filenames, samples = [], split=True):
    """
    Load a bunch of files as a pandas dataframe.
    """
    if not samples:
        samples = [1 for i in range(len(filenames))]

    def Prepare(s):
        s = str(s)
        return ['<S>'] + list(s) + ['</S>']

    df = pd.concat([pd.read_csv(filenames[i], sep='\t').sample(frac = samples[i],
                                                               replace = samples[i]>1,
                                                               random_state = 42)
                    for i in range(len(filenames))])
    if split:
        df['query_'] = df['query'].apply(Prepare)

    return df


class Dataset(object):

    def __init__(self, df, char_vocab, batch_size=24, max_len=50,image_dir={}, image_size = 224, only_char = False):
        self.max_len = max_len
        self.char_vocab = char_vocab
        self.df = df.sample(frac=1)
        self.batch_size = batch_size
        self.current_idx = 0
        self.image_dir = image_dir
        self.image_size = image_size
        self.only_char = only_char
        self.df['lengths'] = self.df.query_.apply(
            lambda x: min(self.max_len, len(x)))

    def GetFeedDict(self, model, channel_mean, random_img = False):
        if self.current_idx + self.batch_size > len(self.df):
            self.current_idx = 0

        idx = range(self.current_idx, self.current_idx + self.batch_size)
        self.current_idx += self.batch_size

        grp = self.df.iloc[idx]
        f1 = np.zeros((self.batch_size, self.max_len))

        grp['images'] = (grp.image_id.astype(str) + '.npy').values
        img_mat = np.zeros((self.batch_size, self.image_size, self.image_size, 3))
        for i in xrange(len(grp)):

            row = grp.iloc[i]
            if random_img:
                img = np.random.rand(self.image_size, self.image_size,3)*255
            else:
                img = np.load(os.path.join(self.image_dir[row['dataset']], row.images))
            img_mat[i,] = img
            for j in range(row.lengths):
                f1[i, j] = self.char_vocab[row.query_[j]]

        img_mat = img_mat - channel_mean
        feed_dict = {
            model.queries: f1,
            model.query_lengths: grp.lengths.values,
            model.images: img_mat,
        }


        return feed_dict