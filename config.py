import logging
log = logging.getLogger('root')
try:
    import numpy as np
    from constants import *
except Exception as e:
    log.critical(e)
    exit(0)

RANDOM_STATE = np.random.randint(10000,25000)

PARAMETERS = {
    'C' : [1e-3 ,1e-2, 1e-1, 1e0, 1e1, 1e2,1e3]
}



CROSS_VALIDATION = CROSS_VALIDATION_MODEL['STRATIFIED_KFOLD']
RESAMPLING = RESAMPLE_MODEL['UNDER_SAMPLING']['RUS']
CRITERION = SCORE['AUC']
TEST_SIZE = 0.25
K_FOLD = [2]