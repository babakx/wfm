import numpy as np
from sklearn.model_selection import train_test_split
from itertools import count
from collections import defaultdict
from scipy.sparse import csr
import random
import sys
import subprocess
import pandas as pd
import os
from data_readers import DataReaders
from collections import OrderedDict

def vectorize_dic(dic, ix=None, p=None):
    """
    Creates a scipy csr matrix from a list of lists (each inner list is a set of values corresponding to a feature)

    parameters:
    -----------
    dic -- dictionary of feature lists. Keys are the name of features
    ix -- index generator (default None)
    p -- dimension of featrure space (number of columns in the sparse matrix) (default None)
    """
    
    if ix is None:
        ix = defaultdict(count(0).__next__)

    n = len(list(dic.values())[0]) # num samples
    g = len(dic.keys())  # num groups
    nz = n * g  # number of non-zeros

    col_ix = np.empty(nz, dtype=int)

    i = 0
    for k, lis in dic.items():
        # append index el with k in order to prevet mapping different columns with same id to same index
        col_ix[i::g] = [ix[str(el) + str(k)] for el in lis]
        i += 1

    row_ix = np.repeat(np.arange(0, n), g)
    data = np.ones(nz)

    if p is None:
        p = len(ix)

    ixx = np.where(col_ix < p)

    return csr.csr_matrix((data[ixx], (row_ix[ixx], col_ix[ixx])), shape=(n, p)), ix


def vectorize_dic_weights(dic, weights_dic, ix=None, p=None):
    """
    Creates a scipy csr matrix from a list of lists (each inner list is a set of values corresponding to a feature)

    parameters:
    -----------
    dic -- dictionary of feature lists. Keys are the name of features
    ix -- index generator (default None)
    p -- dimension of featrure space (number of columns in the sparse matrix) (default None)
    """
    if (ix == None):
        ix = defaultdict(count(0).__next__)

    n = len(list(dic.values())[0])  # num samples
    g = len(dic.keys())  # num groups
    nz = n * g  # number of non-zeros

    col_ix = np.empty(nz, dtype=int)
    data = np.ones(nz)

    i = 0
    feat_g = {}     # Dictionary repsenting the group of feature

    for k, lis in dic.items():
        # append index el with k in order to prevet mapping different columns with same id to same index
        feat_ids = [ix[str(el) + str(k)] for el in lis]
        col_ix[i::g] = feat_ids
        group_features = np.unique(feat_ids)
        group_ix = np.repeat(i, np.size(group_features))
        feat_g.update(dict(zip(group_features, group_ix)))
        data[i::g] *= weights_dic[k]
        i += 1

    row_ix = np.repeat(np.arange(0, n), g)

    if (p == None):
        p = len(ix)

    ixx = np.where(col_ix < p)

    return csr.csr_matrix((data[ixx], (row_ix[ixx], col_ix[ixx])), shape=(n, p)), ix, feat_g

class WfmData():
    def __init__(self, dataset, w_init, m_name, has_context=False, implicit=True):
        self.dr = DataReaders(has_context, w_init, implicit, m_name)
        self.w_init = w_init
        self.dataset = dataset
        self.num_cand = 1000

        if dataset == 'ml1m':
            self.train, self.test, self.train_neg = self.dr.get_movieLens1M()
        elif dataset == 'frappe':
            self.train, self.test, self.train_neg = self.dr.get_frappe()
        elif dataset == 'msd':
            self.train, self.test, self.train_neg = self.dr.get_MSD_T50()
        elif dataset == 'msd20':
            self.train, self.test, self.train_neg = self.dr.get_MSD_T20()
        elif dataset == 'kassandra':
            self.train, self.test, self.train_neg = self.dr.get_kassandr()
        elif dataset == 'goodbooks':
            self.train, self.test, self.train_neg = self.dr.get_goodbooks()

        self.cols = self.dr.cols
        self.weights = OrderedDict(zip(self.dr.cols, self.dr.weights))

        print('preparing data...')
        self.users = set(self.test.UserId).intersection(set(self.train.UserId))
        self.train_items = set(self.train.ItemId)
        self.items = list(self.train_items.intersection(set(self.test.ItemId.unique())))[:self.num_cand]
        
        self.item_attr = None
        ix_col = []
        if len(self.dr.cols) > 2 and dataset != 'frappe':
            self.item_attr = self.train.groupby('ItemId')[self.dr.cols[2]].apply(lambda x: str(list(x)[0]))
        else:
            ix_col = self.cols[2:]

        self.relevant = self.test[self.test.UserId.isin(self.users)].groupby(['UserId'] + ix_col).ItemId.apply(
            lambda x: list(self.train_items.intersection(set(x.values))))

        self.vectorize_data(implicit)
        self.print_stat()

    def vectorize_data(self, implicit):
        print('vectorizing...')
        self.X_train, self.ix, self.gr_train = vectorize_dic_weights(dict(zip(self.cols, self.train[self.cols].values.T)), self.weights)
        self.X_train_neg, _, __ = vectorize_dic_weights(dict(zip(['UserId','ItemId'], self.train_neg[['UserId','ItemId']].values.T)), self.weights, self.ix, self.X_train.shape[1])
        self.X_test, _, __ = vectorize_dic_weights(dict(zip(self.cols, self.test[self.cols].values.T)), self.weights, self.ix, self.X_train.shape[1])
        self.y_train = self.train.Rating.values if 'Rating' in self.train.columns else []
        self.y_test = self.test.Rating.values if 'Rating' in self.test.columns else []
        self.p = self.X_train.shape[1]

    def print_stat(self):
        print ("Train shape: ", self.X_train.shape)
        print ("Test shape: ", self.X_test.shape)
        print ("Train items: ", len(self.train.ItemId.unique()))
        print ("Test items: ", len(self.test.ItemId.unique()))
        print ("Train users: ", len(self.train.UserId.unique()))
        print ("Test users: ", len(self.test.UserId.unique()))

        return self.train, self.test

    def explicit_binary(self):
        self.train['Rating'] = 1
        self.train_neg['Rating'] = 0
        s = pd.concat([self.train, self.train_neg], ignore_index=True)
        s = s.sample(frac=1).reset_index(drop=True)
    
        return s.fillna(1)






