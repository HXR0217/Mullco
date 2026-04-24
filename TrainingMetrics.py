import numpy as np
from sklearn.metrics import hamming_loss # Hamming Loss
from sklearn.metrics import zero_one_loss # One-error
from sklearn.metrics import coverage_error # Coverage
from sklearn.metrics import label_ranking_loss # Ranking Loss
from sklearn.metrics import average_precision_score


epsilon = 1e-8
def example_accuracy(label,predict):
    ex_and = np.sum(np.logical_and(label,predict), axis=1).astype('float32')
    ex_or = np.sum(np.logical_or(label,predict), axis=1).astype('float32')
    return np.mean(ex_and / (ex_or + epsilon))


def evalu_kfold(gt,predict):
    ex_acc = example_accuracy(gt, predict)
    ham_loss = hamming_loss(gt, predict)
    one_error = zero_one_loss(gt, predict,normalize=True)
    cov = coverage_error(gt, predict)
    rank_loss = label_ranking_loss(gt, predict)
    ap = average_precision_score(gt, predict,average='micro')#

    return ex_acc,one_error, cov,rank_loss,ham_loss,ap