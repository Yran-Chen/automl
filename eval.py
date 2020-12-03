# python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/11/19 10:20
# @Author  : Yran CHan
# @Site    : 
# @File    : eval.py
# @Software: PyCharm
# Et0yep

from dataset_pool import DatasetPool
from NNlearner_keras import LFE_learner,imp_net

from common.dataset_transfer import OperatorParser
operatorParser = OperatorParser()

from sklearn import preprocessing
import copy

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

OP_DICT = [
    # 'add','sub','mul','div',
    # 'sqrt','log','square','zscore','sigmoid',
    'square',
]

import numpy as np
import pandas as pd

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

class OptForward(DatasetPool):


    def __init__(self,param):
        super().__init__(param)
        self.operator = param['operator']
        self.lfe_param = param['lfe_param']

    def run(self,**kwargs):
        self.dataset_preprocessing()
        self.eval_forward(**kwargs)

    def run_search(self):
        self.dataset_preprocessing()
        self.eval_brutalsearch()

    def object_labelencoder  (self,data:pd.DataFrame):
        from sklearn import preprocessing
        labelendr = preprocessing.LabelEncoder()

        for idx in data.columns:
            if data[idx].dtype == 'object':
                data[idx] = labelendr.fit_transform(data[idx])

        # self.load_dataset_data(False)

        return  data


    def quantile_labelencoder (self, label, bins=5):
        from common.cal_utils import bins_cut
        label = bins_cut(label,qcut=bins)
        return list(label)

    def eval_forward(self,task):
        from common.cal_utils import quantileSkrechArray
        self.learner = LFE_learner(self.lfe_param)

        for __dataset_name in self.dataset:
            __df_raw_data = copy.deepcopy(self.load_dataset_data(__dataset_name)[0])
            __df_raw_data.iloc[:,0:-1] = self.object_labelencoder(__df_raw_data.iloc[:,0:-1])
            # print (__df_raw_data)

            if task == 'regression':
                original_score = self.run_regression_model(__df_raw_data)
                print('Origin Score: ', original_score)

                __df_data = __df_raw_data
                __df_data.iloc[:,-1] = self.quantile_labelencoder(__df_data.iloc[:,-1])

                for oprtr in self.operator:
                    trans_data = self.cal_data_forward(
                        copy.deepcopy(__df_data),
                        oprtr,
                    )
                    trans_score = self.run_regression_model(trans_data)
                    print('Tran Score:   ', trans_score)

            else :
                original_score = self.run_training_model(__df_raw_data)
                print('Origin Score: ', original_score)

                __df_data = __df_raw_data

                for oprtr in self.operator:
                    trans_data = self.cal_data_forward(
                            copy.deepcopy(__df_data),
                            oprtr,
                            )
                    trans_score = self.run_training_model(trans_data)
                    print ('Tran Score:   ', trans_score)


    def eval_brutalsearch(self,epoch = 2):
        from common.cal_utils import quantileSkrechArray
        self.learner = LFE_learner(self.lfe_param)

        for __dataset_name in self.dataset:
            __df_raw_data = copy.deepcopy(self.load_dataset_data(__dataset_name)[0])
            __df_data = __df_raw_data

            for i in range(epoch):
                for oprtr in self.operator:
                    print (__df_data.shape)
                    __df_data = self.cal_data_forward(__df_data,oprtr)

        return __df_data

    def cal_data_forward(self ,df_data, oprtr,__range=(-10,10)):
        from common.cal_utils import quantileSkrechArray

        __df_trans_data = copy.deepcopy(df_data)
        __df_raw_label = copy.deepcopy(df_data.iloc[:, -1])
        __df_raw_label = __df_raw_label.apply(str)

        """
        Test for origin data.
        """
        # original_score = self.run_training_model(df_data)
        score_deck = []

        for __label in list(__df_raw_label.unique()):
            oprtr_array = []

            __df_trans_label = copy.deepcopy(__df_raw_label)
            __df_trans_label.loc[(__df_raw_label != str(__label))] = 'zzF'
            __df_trans_data.iloc[:, -1] = __df_trans_label

            for __feature in list(__df_trans_data.columns)[0:-1]:
                __df_trans_data.iloc[:, __feature] = operatorParser.feature_trans(oprtr,
                                                                                  __df_trans_data.iloc[:, __feature])

            for __feature in list(__df_trans_data.columns)[0:-1]:
                __QSA_data = np.array(
                    __df_trans_data.iloc[:, __feature].groupby(__df_trans_data.iloc[:, -1]).apply(quantileSkrechArray,
                                                                                                  range=__range)
                ).reshape(-1)

                oprtr_array.append(
                    np.append(__QSA_data, __feature)
                )

            oprtr_array = np.array(oprtr_array)

            y_pred = self.learner.eval(data=oprtr_array[:, 0:-1], oprtr=oprtr).reshape(-1)
            # print (y_pred)
            score_deck.append(y_pred)

        score_deck = np.array(score_deck).mean(axis=0)
        # print(score_deck)
        counter = 0
        for pivoti, score in enumerate(score_deck):
            if score > 0.5:
                counter = counter+1
                df_data = pd.concat([__df_trans_data.iloc[:, pivoti], df_data], axis=1)
                # df_data.iloc[:,pivoti] = __df_trans_data.iloc[:, pivoti]

        # print('tran percent:   ',counter/len(score_deck))
        """
        Test for trans data.
        """
        # trans_score = self.run_training_model(df_data)

        # print('Dataset Name:   ', __dataset_name)
        # print('Operator:       ', oprtr)
        # print("Original score: ", original_score)
        # print("Trans    Score: ", trans_score)
        #
        # print('——' * 40)

        return df_data

    # def eval(self):
    #     self.learner = LFE_learner(self.lfe_param)
    #     for oprtr in self.operator:
    #         tran_data = self.eval_forward(oprtr)

    # def eval_forward(self,oprtr,__range = (-10,10)):
    #     from common.cal_utils import quantileSkrechArray
    #
    #     # print(self.dataset)
    #     for __dataset_name in self.dataset:
    #
    #
    #         __df_raw_data = self.load_dataset_data(__dataset_name)[0]
    #         __df_trans_data =  copy.deepcopy(__df_raw_data)
    #         __df_raw_label = copy.deepcopy(__df_raw_data.iloc[:, -1])
    #         __df_raw_label = __df_raw_label.apply(str)
    #
    #         """
    #         Test for origin data.
    #         """
    #         original_score = self.run_training_model(__df_raw_data)
    #         score_deck = []
    #
    #         for __label in  list( __df_raw_label.unique() ):
    #             oprtr_array = []
    #
    #             __df_trans_label = copy.deepcopy( __df_raw_label )
    #             __df_trans_label.loc[(__df_raw_label != str(__label))] = 'zzF'
    #             __df_trans_data.iloc[:, -1] = __df_trans_label
    #
    #             for __feature in list(__df_trans_data.columns)[0:-1]:
    #                 __df_trans_data.iloc[:, __feature] = operatorParser.feature_trans(oprtr,
    #                                                                                       __df_trans_data.iloc[:, __feature])
    #
    #             for __feature in list(__df_trans_data.columns)[0:-1]:
    #
    #                 __QSA_data = np.array(
    #                 __df_trans_data.iloc[:, __feature].groupby(__df_trans_data.iloc[:, -1]).apply(quantileSkrechArray,range=__range)
    #                 ).reshape(-1)
    #
    #                 oprtr_array.append(
    #                     np.append(__QSA_data,__feature)
    #                 )
    #
    #             oprtr_array = np.array(oprtr_array)
    #
    #             y_pred = self.learner.eval(data=oprtr_array[:,0:-1],oprtr=oprtr).reshape(-1)
    #             # print (y_pred)
    #             score_deck.append(y_pred)
    #
    #         score_deck = np.array(score_deck).mean(axis=0)
    #         print (score_deck)
    #
    #         for pivoti, score in enumerate(score_deck):
    #             if score > 0.5:
    #                 __df_raw_data = pd.concat([__df_trans_data.iloc[:,pivoti] ,__df_raw_data] ,axis=1)
    #
    #         # print (__df_raw_data)
    #         # print (oprtr_array.shape)
    #         # print (__df_raw_data.shape)
    #
    #         # print(y_pred)
    #         # print (__df_raw_data)
    #         """
    #         Test for trans data.
    #         """
    #         trans_score = self.run_training_model(__df_raw_data)
    #
    #         print ('Dataset Name:   ',__dataset_name)
    #         print ('Operator:       ',oprtr)
    #         print ("Original score: ",original_score)
    #         print ("Trans    Score: ",trans_score)
    #
    #         print ('——'*40)
    #
    #
    #         # print (oprtr_array[:,-1])

    def run_regression_model (self,data):
        from sklearn.linear_model import Lasso
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score

        df_raw_data = copy.deepcopy(data)
        data_y = df_raw_data.iloc[:,-1].values
        data_x = df_raw_data.iloc[:,0:-1].values
        # feed into model.
        model = eval(self.param['regression_model_param']['model'])
        clf_svc_cv = model(**self.param['regression_model_param']['model_settings'])

        """
        for fastbacktest/
        """
        clf_svc_cv.fit( X=data_x, y=data_y )
        scores_clf_cv = clf_svc_cv.score( X=data_x, y=data_y)
        # print(scores_clf_cv)
        """
        for cv backtest.
        """
        # scores_clf_cv = cross_val_score(clf_svc_cv, data_x, data_y, cv = 5)
        # print(scores_clf_cv)
        # print("Accuracy: %f (+/- %0.4f)" % (scores_clf_cv.mean(), scores_clf_cv.std() * 2))

        return scores_clf_cv.mean()

    def run_training_model(self,df_raw_data):

        df_raw_data_label = copy.deepcopy( (df_raw_data.iloc[:,-1].apply(str) ) )

        labelendr = preprocessing.LabelEncoder()

        data_y = labelendr.fit_transform(df_raw_data_label)
        data_x = df_raw_data.iloc[:,0:-1].values

        # feed into model.
        model = eval(self.param['model_param']['model'])
        clf_svc_cv = model(**self.param['model_param']['model_settings'])

        """
        for cv backtest.
        """
        scores_clf_cv = cross_val_score(clf_svc_cv, data_x, data_y, cv = 5)
        # print(scores_clf_cv)
        # print("Accuracy: %f (+/- %0.4f)" % (scores_clf_cv.mean(), scores_clf_cv.std() * 2))

        return scores_clf_cv.mean()



if __name__ == '__main__':



    param_demo = {

    'name':'1130_test',
    'data_dir': r'D:\!DTStack\Dataset\UCI_\ml\machine-learning-databases',
    'save_dir': r"D:\!DTStack\Savefile_remote",
    'operator':['stdscaler', 'zscore', 'sigmoid','cbrt','freq'],
    'selected': '!reg_00332',
    'model_param': {
        'model': 'LogisticRegression',
        "model_settings": {
            'max_iter': 3000,
        },
    },

    # 'regression_model_param':{
    #     'model': 'LinearRegression',
    #     "model_settings": {
    #         # 'alpha': 0.1,
    #         # 'max_iter': 3000,
    #     },
    # },
    'regression_model_param':{
        'model': 'Lasso',
        "model_settings": {
            'alpha': 0.1,
            'max_iter': 10000,
        },
    },

    'lfe_param':   {
                'name': 'test_beta',
                'data_dir': r"D:\!DTStack\Savefile_remote",
                'eval_data_dir': r'D:\!DTStack\Savefile_remote\dataset\test_beta_eval_00489',
                'train_name': 'demo_model_balanced',
                'operator': None,
                'model': imp_net,
                'if_balanced': True,
                'balance_percent': 0.5005,
                'balance_method': 'downsample',
                'epoch': 300,
                },
}

    __opt = OptForward(param_demo)
    # __opt.run_search()
    __opt.run(task='regression')

    print('Demo.')