import numpy as np
import sklearn


def tpr_at_fpr(scores, labels, fpr_rate) -> float:
    """Computes the true positive rate @ a given false positive rate"""
    c_fpr = np.quantile(scores[labels == 0], 1 - fpr_rate)
    fpr = (scores[labels == 0] >= c_fpr).mean()
    tpr = (scores[labels == 1] >= c_fpr).mean()
    return fpr, tpr


def confusion_matrix_at_fpr(scores, labels, fpr_rate):
    """Computes confusion matrix @ a given false positive rate"""
    c_fpr = np.quantile(scores[labels == 0], 1 - fpr_rate)
    scores[scores >= c_fpr] = 1
    scores[scores < c_fpr] = 0
    return sklearn.metrics.confusion_matrix(labels, scores)
