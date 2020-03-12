import os
os.system("pip3 install virtualenv")
os.system("virtualenv -p `which python3` envLogR")
os.system("source envLogR/bin/activate")
os.system("pip3 install -r requirements.txt")
# Disable warning sklearn
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import log as logger
log = logger.setup_custom_logger('root')
try:
    import pandas as pd
    import numpy as np
    from config import *
    from constants import SCORE
    from v1.LogisticRegressionModel import Model as LR
    from sklearn.model_selection import train_test_split
except Exception as e :
    os.system('deactivate')
    log.critical(e)
    exit(0)

try:
    dataset = pd.read_hdf("./dataset/dataset.h5",key='dataset')

    LR = LR(PARAMETERS)
    X, y = LR.Preprocessing(dataset)

    X_train_validate, X_test, y_train_validate, y_test = train_test_split(X,
                                                                        y, 
                                                                        test_size=TEST_SIZE,
                                                                        stratify=y,
                                                                        random_state=RANDOM_STATE)

    LR.fit(X_train_validate,y_train_validate)

    log.info(f'best score: {LR.best_score_}')
    log.info(f'best model: {LR.get_best_estimator()}')

    y_predict = LR.predict(X_test)

    auc_score = SCORE['AUC'](y_test,y_predict)
    accuracy_score = SCORE['LOSS_LOG'](y_test,y_predict)
    sensitivity_score = SCORE['SENSITIVITY'](y_test,y_predict)
    specificity_score = SCORE['SPECIFICITY'](y_test,y_predict)
    fpr_score = SCORE['FPR'](y_test,y_predict)
    g_mean_score = SCORE['G_MEAN'](y_test,y_predict)

    log.info(f'''
    accuracy    = {accuracy_score}
    auc         = {auc_score}
    sensitivity = {sensitivity_score}
    specificity = {specificity_score}
    fpr         = {fpr_score}
    g-mean      = {g_mean_score}''')
    
except Exception as e :
    os.system('deactivate')
    log.error(e)

os.system('deactivate')
