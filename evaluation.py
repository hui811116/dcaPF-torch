import numpy as np
import os
import sys
import utils as ut
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, v_measure_score, accuracy_score

SEED_DISCRETE=123 #For reproduction of the discrete dataset

def samplingDiscrete(pxy,nsample):
    rng = np.random.default_rng(seed=SEED_DISCRETE)
    (px,py,pxcy,pycx) = ut.priorInfo(pxy)
    # bayes decoder
    y_prob = rng.random((nsample,))
    y_map = np.cumsum(py)
    y_samp = []
    for idx,yp in enumerate(y_prob):
        for yi,ym in enumerate(y_map):
            if yp < ym:
                y_samp.append(yi)
                break
    y_samp = np.array(y_samp)
    x_samp = []
    x_prob = rng.random((nsample,))
    x_map = np.cumsum(pxcy,axis=0)
    for xi, xp in enumerate(x_prob):
        for xd, xm in enumerate(x_map[:,y_samp[xi]]):
            if xp < xm:
                x_samp.append(xd)
                break
    x_samp = np.array(x_samp)
    return x_samp, y_samp

def evalBayes(pzcx,pxy,nsample):
    (px,py,pxcy,pycx) = ut.priorInfo(pxy)
    # bayes decoder
    pz = np.sum(pzcx * px[None,:],axis=1)
    pzy = pzcx @ pxy
    pycz = (pzy/np.sum(pzy,axis=1,keepdims=True)).T
    
    pxcz = ((pzcx * px[None,:])/pz[:,None]).T
    x_sample, y_label = samplingDiscrete(pxy,nsample)
    z_sample =bayesSampling(pzcx,x_sample)
    x_hat_sample = bayesSampling(pxcz,z_sample)
    y_hat_sample = bayesSampling(pycz,z_sample)
    # label matching...
    x_acc, _, _ = clustering_accuracy(x_sample,x_hat_sample)
    y_acc,_ ,_  = clustering_accuracy(y_label,y_hat_sample)
    return x_acc, y_acc

def bayesSampling(pzcx,xsamples):
    rng = np.random.default_rng()
    z_prob = rng.random((len(xsamples),))
    z_sample =[]
    zmaps = np.cumsum(pzcx,axis=0)
    for idx, xs in enumerate(xsamples):
        tmp_map = zmaps[:,xs]
        for zi, zm in enumerate(tmp_map):
            
            if zm > z_prob[idx]:
                z_sample.append(zi)
                break
    return np.array(z_sample)

def compute_metrics(y_true,y_pred):
	out_nmi = nmi(y_true,y_pred)
	out_ari = ari(y_true,y_pred)
	out_vms = vmeasure(y_true,y_pred)
	out_pur = purity(y_true,y_pred)
	out_acc, out_acc_cnt, out_total_cnt = clustering_accuracy(y_true,y_pred)
	return {
		"nmi":out_nmi,
		"ari":out_ari,
		"vmeasure":out_vms,
		"purity":out_pur,
		"acc":out_acc,
		"acc_cnt":int(out_acc_cnt),
		"total_cnt":out_total_cnt,
	}

def nmi(y_true, y_pred):
	return normalized_mutual_info_score(y_true,y_pred)
def ari(y_true, y_pred):
	return adjusted_rand_score(y_true,y_pred)
def vmeasure(y_true, y_pred):
	return v_measure_score(y_true, y_pred)

def purity(y_true, y_pred):
    """Purity score
        Args:
            y_true(np.ndarray): n*1 matrix Ground truth labels
            y_pred(np.ndarray): n*1 matrix Predicted clusters

        Returns:
            float: Purity score
    """

    # matrix which will hold the majority-voted labels
    y_voted_labels = np.zeros(y_true.shape)
    # Ordering labels
    ## Labels might be missing e.g with set like 0,2 where 1 is missing
    ## First find the unique labels, then map the labels to an ordered set
    ## 0,2 should become 0,1
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that 
    # we count the actual occurence of classes between two consecutive bins
    # the bigger being excluded [bin_i, bin_i+1]
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner

    return accuracy_score(y_true, y_voted_labels)

def clustering_accuracy(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    acc_cnt = sum([w[i, j] for i, j in ind]) * 1.0
    total_cnt = y_pred.size
    return acc_cnt / y_pred.size, acc_cnt, total_cnt


def match(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    new_y = np.zeros(y_true.shape[0])
    for i in range(y_pred.size):
        for j in row_ind:
            if y_true[i] == col_ind[j]:
                new_y[i] = row_ind[j]
    #new_y = torch.from_numpy(new_y).long().to(device)
    #new_y = new_y.view(new_y.size()[0])
    return new_y