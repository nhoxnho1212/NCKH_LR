import logging
log = logging.getLogger('root')
try:
    import numpy as np
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.combine import SMOTETomek, SMOTEENN
    from imblearn.under_sampling import RandomUnderSampler, NearMiss
    from sklearn.model_selection import StratifiedKFold, KFold
    from util.score import *
    from datetime import datetime
except Exception as e:
    log.critical(e)
    exit(0)

SCORE = {
    'RECALL' : recall,
    'ACCURACY': accuracy,
    'F1' : f1,
    'AUC': auc,
    'SENSITIVITY' : sensivity,
    'SPECIFICITY' : specificity,
    'FPR' : fpr,
    'G_MEAN' : g_mean,
    'LOG_LOSS' : logLoss
}

''' 
    Under sampling :
        - RUS = RandomUnderSanpler
        - NM = NearMiss
    Over Sampling :
        - ROS = RandomOverSampler
        - SMOTE = SMOTE

'''
RESAMPLE_MODEL = {
    'UNDER_SAMPLING' : {
        'RUS' : RandomUnderSampler,
        'NM' : NearMiss
    },
    'OVER_SAMPLING' : {
        'SMOTE' : SMOTE,
        'ROS' : RandomOverSampler
    },
    'HYBRID' : {
        'SMOTE_TOMK' : SMOTETomek,
        'SMOTE_ENN' : SMOTEENN
    }
}
CROSS_VALIDATION_MODEL = {
    'KFOLD' : KFold,
    'STRATIFIED_KFOLD': StratifiedKFold
}

FILE_LOG =f'./log/debug_{datetime.now().strftime("%Y-%m-%d")}.log'

