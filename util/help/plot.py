
import logging
log = logging.getLogger('root')
try:
    from sklearn.metrics import roc_curve, auc, plot_roc_curve
    import matplotlib.pyplot as plt
    from itertools import cycle
    from constants import SCORE
    cycol = cycle('bgrcmk')
except Exception as e:
    log.critical(e)


'''
    return (ax, roc_auc, fpr, tpr)
'''
def plot_roc(ax = None, y_true = None, y_predict = None, name = None,score_model = SCORE['AUC']):
    try:
        assert ax != None , 'ax is null'
        assert (not y_true.isnull().any().any()) or y_true != None , 'y_true is null'
        assert (not y_true.isnull().any().any()) or y_predict != None , 'y_predict is null'
        assert name != None , 'name is null'
        fpr, tpr, thresholds = roc_curve(y_true, y_predict, pos_label = 1)
        score = score_model(y_true, y_predict)
        ax.plot(fpr, tpr, color=next(cycol),
            label=f'{name} ({score_model.__name__} = {score:0.4})',
            lw=1, alpha=.3)
            
        return (ax, score, fpr, tpr)
    except Exception as e:
        log.error(e)
        return None
