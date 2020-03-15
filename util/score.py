import os
import logging
log = logging.getLogger('root')
try:
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score, roc_curve, log_loss
    from sklearn.metrics import auc as roc_auc
    import numpy as np
except Exception as e :
    log.critical(e)
    exit(0)

try:
    from config import THRESHOLD
except :
    THRESHOLD = 0.5

def accuracy(y_true,y_predict):
    try:
        y_predict = y_predict >= THRESHOLD
        return accuracy_score(y_true,y_predict)
    except Exception as e:
        log.error(e)

def f1(y_true,y_predict):
    try:
        y_predict = y_predict >= THRESHOLD
        return f1_score(y_true,y_predict)
    except Exception as e:
        log.error(e)

def recall(y_true,y_predict):
    try:
        y_predict = y_predict >= THRESHOLD
        return recall_score(y_true,y_predict)
    except Exception as e:
        log.error(e)

def auc(y_true,y_predict):
    try:
        fpr, tpr, thresholds = roc_curve(y_true,y_predict, pos_label = 1)
        return roc_auc(fpr, tpr)
    except Exception as e:
        log.error(e)

def g_mean(y_true,y_predict):
    try:
        y_predict = y_predict >= THRESHOLD
        (TN, FP, FN, TP) = confusion_matrix(y_true,y_predict).ravel()
        return np.sqrt((TP/(TP + FN)) * (TN/(TN + FP)))
    except Exception as e:
        log.error(e)

def sensivity(y_true,y_predict):
    try:
        y_predict = y_predict >= THRESHOLD
        (TN, FP, FN, TP )=confusion_matrix(y_true,y_predict).ravel()
        return TP/(TP+FN)
    except Exception as e:
        log.error(e)

def specificity(y_true,y_predict):
    try:
        y_predict = y_predict >= THRESHOLD
        (TN, FP, FN, TP)=confusion_matrix(y_true,y_predict).ravel()
        return TN/(TN+FP)
    except Exception as e:
        log.error(e)

def fpr(y_true,y_predict):
    try:
        y_predict = y_predict >= THRESHOLD
        (TN, FP, FN, TP)=confusion_matrix(y_true,y_predict).ravel()
        return FP/(TN+FP)
    except Exception as e:
        log.error(e)

def logLoss(y_true,y_predict):
    try:
        return log_loss(y_true,y_predict)
    except Exception as e:
        log.error(e)