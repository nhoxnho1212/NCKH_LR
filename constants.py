import logging
log = logging.getLogger('root')
try:
    import numpy as np
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler, TomekLinks
    from sklearn.model_selection import StratifiedKFold, KFold
    from util.score import *
    from datetime import datetime
except Exception as e:
    log.critical(e)
    exit(0)

SCORE = {
    'RECALL' : recall,
    'LOSS_LOG': accuracy,
    'F1' : f1,
    'AUC': auc,
    'SENSITIVITY' : sensivity,
    'SPECIFICITY' : specificity,
    'FPR' : fpr,
    'G_MEAN' : g_mean
}

''' 
    Under sampling :
        - RUS = RandomUnderSanpler
        - Tomk = TomkLinks
    Over Sampling :
        - SMOTE = SMOTE
'''
RESAMPLE_MODEL = {
    'UNDER_SAMPLING' : {
        'RUS' : RandomUnderSampler,
        'Tomk' : TomekLinks
    },
    'OVER_SAMPLING' : {
        'SMOTE' : SMOTE
    }
}
CROSS_VALIDATION_MODEL = {
    'KFOLD' : KFold,
    'STRATIFIED_KFOLD': StratifiedKFold
}

FILE_LOG =f'./log/debug_{datetime.now().strftime("%Y-%m-%d")}.log'

