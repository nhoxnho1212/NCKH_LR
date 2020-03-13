import logging
log = logging.getLogger('root')
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import MinMaxScaler,LabelEncoder
    from sklearn.model_selection import train_test_split
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import pickle
    from util.help.featurePreprocessing import oneHotProcessing
    from util.help.plot import plot_roc
    from util import initFolder
    from config import *
except Exception as e:
    log.critical(e)
    exit(0)

class Model:
    '''
    Dataset is Lending club dataset
    Training dataset with model Logistic Regression.
    Pipeline:
        - Preprocessing
        - Training with stratifiedKFold
            - Resampling
            - Trainning
        - Fitting and Tuning
    
    Attribute:
        - params
        - boolean_features
        - numeric_feature
        - categorical_features_onehot
        - categorical_feature_labelencode
        - best_score_
    Method:
        - set_params : set parameters for Logistic regression model
            - params : parameters
        - Preprocessing : 
            - dataset : dataframe
        - fit : fitting and tuning parameters
            - X_train_validate
            - y_train_validate
            - save (True / False) : default is True : save model to directory result/model
        - get_best_estimator : get best model
            return model sklearn logistic regression
        - predict : 
            - X_predict
            return y_predict
    '''
    def __init__(self,params = None):
        try :
            assert 'C' in params, 'set parameters fail: params do not have C'
            self.params = dict()
            self.params['C'] = params['C']
            self.__LogR = LogisticRegression(random_state=RANDOM_STATE)
            self.boolean_features = ['verification_status','pymnt_plan','hardship_flag','debt_settlement_flag']
            self.numeric_feature = ['loan_amnt', 
                   'int_rate', 
                   'annual_inc', 
                   'dti', 
                   'delinq_2yrs', 
                   'fico_range_low', 
                   'inq_last_6mths', 
                   'open_acc', 
                   'pub_rec',
                   'revol_bal', 
                   'revol_util',
                   'total_acc', 
                   'tot_cur_bal', 
                   'chargeoff_within_12_mths',
                   'delinq_amnt', 
                   'pub_rec_bankruptcies']
            self.categorical_features_onehot=['term','home_ownership']
            self.categorical_feature_labelencode = ['purpose','addr_state']
            self.__best_estimator_ = pickle.dumps(self.__LogR)
            self.best_score_ = 0.0 
        except Exception as e :
            log.error(e)

    def set_params(self,params = None):
        try:
            assert 'C' in params, 'set parameters fail: do not have C'
            self.params['C'] = params['C']
            log.info('set parameters success!')
        except Exception as e :
            log.error(e)

    def __label_encoder_grade(self,x):
        if x == 'A':
            return 0
        elif x == 'B':
            return 1
        elif x == 'C':
            return 2
        elif x == 'D':
            return 3
        elif x == 'E':
            return 4
        elif x == 'F':
            return 5
        elif x == 'G':
            return 6

    ''' 
    return (X, y)
    '''
    def Preprocessing(self,dataset):
        try :
            # assert dataset != None , 'dataset is null'
            for col in self.boolean_features + self.categorical_feature_labelencode + self.categorical_features_onehot + self.numeric_feature:
                assert col in dataset , f'dataset do not column name: {col}'
            
            # Nummeric features
            dataset[self.numeric_feature] = MinMaxScaler().fit_transform(dataset[self.numeric_feature])
            
            # Ordinal categorical feature
            dataset.grade = dataset.grade.apply(self.__label_encoder_grade)

            # Boolean features
            for f in self.boolean_features:
                dataset[f] = LabelEncoder().fit_transform(dataset[f])

            # Categorical features less than 10 unique values
            for f in self.categorical_features_onehot:
                dataset=oneHotProcessing(f,dataset)

            # Categorical features more than 10 unique values

            for f in self.categorical_feature_labelencode:
                dataset[f] = LabelEncoder().fit_transform(dataset[f])

            X = dataset.drop(columns=['loan_condition'])

            y = dataset['loan_condition']

            return (X, y)

        except Exception as e :
            log.error(e)
            print('Preprocessing fail !')
            return (None, None)

    def fit(self, X_train_validate = None, y_train_validate = None,save = True): 
        C_best = 0
        KFold_best = 0
        iter_split_best = 0
        try :
            assert (not X_train_validate.isnull().any().any()) or X_train_validate != None, 'X_train is null'
            assert (not y_train_validate.isnull().any().any()) or y_train_validate != None, 'y_train is null'
            pathNameSave = './'
            initFolder.init(f'{pathNameSave}result')
            initFolder.init(f'{pathNameSave}result/model')
            initFolder.init(f'{pathNameSave}result/ROC')
            for C in self.params['C']:
                print(f'C = {C}')
                for k_split in K_FOLD:
                    print(f'-- K_FOLD = {k_split}')
                    kf = CROSS_VALIDATION(n_splits=k_split,random_state=RANDOM_STATE)

                    tprs = []
                    scores = []
                    mean_fpr = np.linspace(0, 1, 100)

                    fig, ax = plt.subplots()

                    for i, (train_index, validate_index) in enumerate(kf.split(X_train_validate,y_train_validate)):
                        print(f'-- -- k = {i} done :',end='')
                        X_train, X_validate = X_train_validate.iloc[train_index], X_train_validate.iloc[validate_index]
                        
                        y_train, y_validate = y_train_validate.iloc[train_index], y_train_validate.iloc[validate_index]

                        #resample (imbalance processing)
                        try:
                            rs = RESAMPLING(random_state=RANDOM_STATE)
                        except :
                            rs = RESAMPLING()
                        X_train_rs, y_train_rs = rs.fit_sample(X_train, y_train)

                        X_validate_rs,y_validate_rs = rs.fit_resample(X_validate, y_validate)
                        print(f'resamling',end=', ')
                        
                        #train
                        self.__LogR.set_params(C = C)
                        self.__LogR.fit(X_train_rs,y_train_rs)
                        print(f'training',end=', ')
                        if save :
                            #save model to disk
                            filename = f'{pathNameSave}result/model/LR_{C}_{i}_{k_split}_{RANDOM_STATE}_{RESAMPLING.__name__}.sav'
                            pickle.dump(self.__LogR, open(filename, 'wb'))
                            print(f'saving model')

                        y_predict_valid = self.__LogR.predict_proba(X_validate_rs)
                        
                        (ax, score, fpr, tpr) = plot_roc(ax,y_validate_rs,y_predict_valid[:,1],f'ROC fold {i}',CRITERION)
                        
                        interp_tpr = np.interp(mean_fpr, fpr, tpr)
                        interp_tpr[0] = 0.0
                        tprs.append(interp_tpr)
                        scores.append(score)

                        # Find largest score 
                        if self.best_score_ < score :
                            self.best_score_ = score
                            C_best = C
                            KFold_best = k_split
                            iter_split_best = i
                            self.__best_estimator_ = pickle.dumps(self.__LogR)
                            

                    mean_tpr = np.mean(tprs, axis=0)
                    mean_tpr[-1] = 1.0
                    mean_score = np.mean(scores)
                    std_score = np.std(scores)

                    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                            label='Chance', alpha=.8)
                    ax.plot(mean_fpr, mean_tpr, color='b',
                            label=f'Mean ROC ({CRITERION.__name__} = {mean_score:0.4} ± {std_score:0.4})',
                            lw=2, alpha=.8)


                    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
                        title=f"ROC Logistic Regression (c = {C}, {RESAMPLING.__name__})")
                    ax.legend(loc="lower right")
                    plt.savefig(f'{pathNameSave}result/ROC/LR_ROC_{C}_{RANDOM_STATE}_{k_split}_{RESAMPLING.__name__}.png')

                log.info(f'train succesed: c = {C}')
            log.info('train sucessed!!')
            
            # Save best file to result/
            if save :
                filename = f'{pathNameSave}result/LR_{C_best}_{iter_split_best}_{KFold_best}_{RANDOM_STATE}_{RESAMPLING.__name__}.sav'
                pickle.dump(filename, open(filename, 'wb'))

        except Exception as e :
            log.error(e)

    def get_best_estimator(self):
        return pickle.loads(self.__best_estimator_)

    def predict(self,X_predict): 
        model = pickle.loads(self.__best_estimator_)
        y_predict = model.predict_proba(X_predict)[:,1]

        return y_predict

    