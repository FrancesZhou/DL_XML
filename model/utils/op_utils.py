'''
Created on Nov, 2017

@author: FrancesZhou
'''

from __future__ import absolute_import
import numpy as np


# if len(np.intersect1d(pre_labels, true_labels)):
#     count += 1
# return count*1.0/num
# return count, count * 1.0 / num

def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))

def ndcg_at_k(r, k, true_num=5):
    #dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    dcg_max = dcg_at_k(np.ones(k), min(k, true_num))
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max

def results_for_prop_vector(tar_pid_label_prop, pre_pid_label_prop):
    wts_p_1 = []
    wts_p_3 = []
    wts_p_5 = []
    wts_ndcg_1 = []
    wts_ndcg_3 = []
    wts_ndcg_5 = []
    for pid, true_label_prop in tar_pid_label_prop.items():
        # for propensity loss
        wts_r = pre_pid_label_prop[pid]
        opt_r = sorted(true_label_prop, reverse=True)
        if len(opt_r) < 5:
            opt_r = opt_r + [0]*(5-len(opt_r))
        wts_p_1.append(np.mean(wts_r[:1]) / np.mean(opt_r[:1]))
        wts_p_3.append(np.mean(wts_r[:3]) / np.mean(opt_r[:3]))
        wts_p_5.append(np.mean(wts_r[:5]) / np.mean(opt_r[:5]))
        wts_ndcg_1.append(ndcg_at_k(wts_r, 1, 1) / ndcg_at_k(opt_r, 1, 1))
        wts_ndcg_3.append(ndcg_at_k(wts_r, 3, 3) / ndcg_at_k(opt_r, 3, 3))
        wts_ndcg_5.append(ndcg_at_k(wts_r, 5, 5) / ndcg_at_k(opt_r, 5, 5))
    return np.mean([wts_p_1, wts_p_3, wts_p_5, wts_ndcg_1, wts_ndcg_3, wts_ndcg_5], axis=1)
