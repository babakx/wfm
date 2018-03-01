import tensorflow as tf
from tffm import TFFMRegressor
from tqdm import tqdm
import numpy as np
import time
import os
from scipy.sparse import csr
from itertools import count
from collections import defaultdict
import pandas as pd
from tffm import utils as ut
from functools import reduce

class WfmModel:
    def __init__(self, wfm_data, m_name, order, k, bs, lr, init, reg):
        self.data = wfm_data

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        #session_config = tf.ConfigProto(log_device_placement=False, device_count={'GPU': 1})

        self.lr = lr
        self.num_cand = 1000
        self.m_name = m_name

        self.model = TFFMRegressor(
            order=order,
            rank=k,
            optimizer=tf.train.GradientDescentOptimizer(learning_rate=lr),
            session_config=tf.ConfigProto(log_device_placement=False, device_count={'GPU': 1}),
            n_epochs=1,
            batch_size=bs,
            init_std=init,
            reg=reg,
            input_type='sparse',
            seed=42,
        )

        if m_name == 'bpr':
            self.model.core.mf = True

        if m_name == 'bpr' or m_name == 'fmp' or m_name == 'wfmp':
            self.model.core.loss_function = ut.loss_bpr

        if m_name == 'wfm' or m_name == 'wfmp':
            self.model.core.G = list(self.data.gr_train.values())
            self.model.core.gamma_init = np.array(self.data.dr.weights).astype(np.float32)
            if self.data.w_init == 'all-one' or self.data.w_init == 'all-diff':
                self.model.core.M = np.repeat(True, len(self.data.dr.weights))
            else:
                self.model.core.M = np.append([False, False], np.repeat(True, len(self.data.dr.weights) - 2))

        fit_methods = {'bpr': 'fit_bpr', 'fm': 'fit', 'fmp': 'fit_bpr', 'wfm': 'fit', 'wfmp': 'fit_bpr', 'wmf': 'fit'}

        self.fit_method = fit_methods[m_name]
        self.c = None
        if m_name == 'wmf':
            self.c = self.data.dr.c
            self.model.core.has_conf = True

        print('preparing test matrix...')
        if self.data.dataset == 'frappe':
            self.X_test = self.get_test_matrix_opr()
        else:
            self.X_test, self.rel_c = self.get_test_matrix_ub()


    def get_test_matrix_opr(self):
        items = self.data.items
        relevant = self.data.relevant
        c_cols = self.data.cols[2:]
        nc = self.num_cand
        n = relevant.apply(lambda x: len(x) * (nc + 1)).sum()
        ix = self.data.ix

        i_ix = np.zeros(n, dtype=np.int32)
        u_ix = np.zeros(n, dtype=np.int32)
        c_ix = {}
        for c in c_cols:
            c_ix[c] = np.zeros(n, dtype=np.int32)

        l = 0
        for kk in relevant.keys():
            cands = np.random.choice(np.setdiff1d(items, relevant[kk]), nc)
            for i in relevant[kk]:
                u_ix[l:l + nc + 1] = np.repeat(ix[str(kk[0] if len(c_cols) > 0 else kk) + 'UserId'], nc + 1)
                i_ix[l:l + nc] = [ix[str(ii) + 'ItemId'] for ii in cands]
                i_ix[l + nc] = ix[str(i) + 'ItemId']
                for ii, c in enumerate(c_cols):
                    c_ix[c][l:l + nc + 1] = np.repeat(ix[str(kk[ii + 1]) + c], nc + 1)
        
                l += nc + 1
        
        g = len(c_cols) + 2

        data_m = np.ones(n*g,dtype=bool)
        row_ix = np.repeat(np.arange(0, n, dtype=np.int32), g)
        col_ix = np.zeros(n*g, dtype=np.int32)

        col_ix[0::g] = u_ix
        col_ix[1::g] = i_ix

        for ii, c in enumerate(c_cols):
            col_ix[ii+2::g] = c_ix[c]

        p = self.data.p
        X = csr.csr_matrix((data_m, (row_ix, col_ix)), shape=(n, p))

        return X

    def get_test_matrix_ub(self):
        items = self.data.items
        users = self.data.users
        relevant = self.data.relevant
        c_cols = self.data.cols[2:]
        nc = self.num_cand
        n = relevant.apply(lambda x: len(x) + nc).sum()
        ix = self.data.ix

        item_attr = {}
        if self.data.item_attr is not None:
            item_attr = dict(self.data.item_attr)
            c_ix_ = [ix[str(item_attr[i]) + c_cols[0]] for i in items]
            c_ix = np.zeros(n, dtype=np.int32)
        
        i_ix_ = [ix[str(i) + 'ItemId'] for i in items]

        i_ix = np.zeros(n, dtype=np.int32)
        u_ix = np.zeros(n, dtype=np.int32)

        rel_c = []
        l = 0
        for u in users:
            r = np.size(relevant[u])
            u_ix[l:l + nc + r] = np.repeat(ix[str(u) + 'UserId'], nc + r)
            i_ix[l:l + nc] = i_ix_
            i_ix[l + nc: l + nc + r] = [ix[str(i) + 'ItemId'] for i in relevant[u]] 
            if self.data.item_attr is not None:
                c_ix[l:l + nc] = c_ix_
                c_ix[l + nc:l + nc + r] = [ix[str(item_attr[i]) + c_cols[0]] for i in relevant[u]]
            l += nc + r
            rel_c.append(nc + r)

        g = len(c_cols) + 2

        data_m = np.ones(n*g,dtype=bool)
        row_ix = np.repeat(np.arange(0, n, dtype=np.int32), g)
        col_ix = np.zeros(n*g, dtype=np.int32)

        col_ix[0::g] = u_ix
        col_ix[1::g] = i_ix
        if self.data.item_attr is not None:
            col_ix[2::g] = c_ix

        p = self.data.p
        X = csr.csr_matrix((data_m, (row_ix, col_ix)), shape=(n, p))

        return X, rel_c

    def calc_metrics_opr(self, pred, k):
        relevant = self.data.relevant
        nc = self.num_cand
        hit_counts = []
        rrs = []
        l = 0
        for kk in relevant.keys():
            for i in relevant[kk]:
                top_ix = np.argpartition(pred[l:l+nc + 1], -k)[-k:]
                hit_count = len(np.where(top_ix >= nc)[0])  
                hit_counts.append(hit_count)

                top_val = pred[l + top_ix]
                top_ix = map(lambda x: x[0], sorted(zip(top_ix, top_val), key=lambda x: x[1], reverse=True))

                rr = 0
                for j, item_ix in enumerate(top_ix):
                    if (item_ix >= nc):  #if item is relavant
                        rr = 1 / (j + 1)
                        break;
                rrs.append(rr)
                l += nc + 1
    
        recall = np.sum(hit_counts) / np.size(hit_counts)
        mrr = np.mean(rrs)

        return recall, mrr, recall / k

    def calc_metrics_ub(self, pred, k, rel_c):
        nc = self.num_cand
        hit_counts = []
        recalls = []
        rrs = []
        l = 0
        for c in rel_c:
            top_ix = np.argpartition(pred[l:l+c], -k)[-k:]
            hit_count = len(np.where(top_ix >= nc)[0]) 
            hit_counts.append(hit_count)
            recalls.append(hit_count / (c - nc) if c > nc else 0)

            top_val = pred[l + top_ix]
            top_ix = map(lambda x: x[0], sorted(zip(top_ix, top_val), key=lambda x: x[1], reverse=True))
    
            rr = 0
            for j, item_ix in enumerate(top_ix):
                if (item_ix >= nc):  #if item is relavant
                    rr = 1 / (j + 1)
                    break;
            rrs.append(rr)
            l += c
    
        prc = np.sum(hit_counts) / (k * np.size(hit_counts))
        recall = np.mean(recalls)
        mrr = np.mean(rrs)

        return recall, mrr, prc

    def eval_model(self):
        if self.data.dataset == 'frappe':
            pred = self.model.predict(self.X_test, pred_batch_size=100000)
            r5, mrr5, prc5 = self.calc_metrics_opr(pred, 5)
            r10, mrr10, prc10 = self.calc_metrics_opr(pred, 10)
            r20, mrr20, prc20 = self.calc_metrics_opr(pred, 20)
        else:
            pred = self.model.predict(self.X_test, pred_batch_size=1000000)
            r5, mrr5, prc5 = self.calc_metrics_ub(pred, 5, self.rel_c)
            r10, mrr10, prc10 = self.calc_metrics_ub(pred, 10, self.rel_c)
            r20, mrr20, prc20 = self.calc_metrics_ub(pred, 20, self.rel_c)
        
        return r5, r10, r20, mrr5, mrr10, mrr20, prc5, prc10, prc20

    def train_model(self, epochs, eval_freq, eval_file=None):
        writer = None
        if eval_file is not None:
            writer = open(eval_file, 'w')
            writer.write('Method,WeightInit,Context,Epoch,Order,K,BatchSize,LearnRate,InitStd,Reg,Recall@5,Recall@10,Recall@20,MRR@5,MRR@10,MRR@20,Precision@5,Precision@10,Precision@20,EpochTime,EvalTime,Weights,NewEval,Optimizer,MsdContext,NormalizeAlpha\n')

        def eval_epoch(ep_, epoch_time_):
            start_time = time.time()
            r5, r10, r20, mrr5, mrr10, mrr20, prc5, prc10, prc20 = self.eval_model()
            eval_time = time.time() - start_time
            if self.model.core.G is not None:
                ws = reduce(lambda x, y: str(x) + ' ' + str(y), self.model.session.run(self.model.core.alpha))
            else:
                ws = 'NA'

            writer.write('{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18},{19},{20},{21},{22},{23},{24},{25}\n'.format(
                         self.m_name, self.data.w_init, self.data.dr.context, ep_, self.model.core.order, self.model.core.rank, self.model.batch_size,
                         self.lr, self.model.core.init_std, self.model.core.reg,
                         r5, r10, r20, mrr5, mrr10, mrr20, prc5, prc10, prc20, epoch_time_, eval_time, ws,'True2','GD','Genre',self.model.core.norm_alpha))
            writer.flush()

        total_time = 0
        for ep in tqdm(range(epochs), unit='epoch'):
            start_time = time.time()
            if self.fit_method == 'fit':
                self.model.fit(self.data.X_train, self.data.y_train, c_=self.c)
            else:
                self.model.fit_bpr(self.data.X_train, self.data.X_train_neg)

            epoch_time = time.time() - start_time
            total_time += epoch_time

            if (ep + 1) % eval_freq == 0:
                eval_epoch(ep + 1, epoch_time)

        if writer is not None:
            writer.close()

