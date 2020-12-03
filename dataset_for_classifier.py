import numpy as np
import pandas as pd
import os
import json
import codecs
import pickle
from time import *

from sklearn import preprocessing
import copy
from multiprocessing import Process, Pool

model_param = {
    'model':'GradientBoostingClassifier',

    "model_settings": {
        "loss": "deviance",
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3,
        "min_samples_leaf": 1,
        "subsample": 1.0,
        "max_features": 0.8,
        "validation_fraction": 0.1,
        "random_state": 42
    }
}

OP_DICT = [
    # 'add','sub','mul','div',
    # 'sqrt','log','square','zscore','sigmoid',
    'square',
]

PARAM_TEST = {
    'name':'test',
    'data_dir': r'D:\!DTStack\Dataset\UCI_\ml\machine-learning-databases',
    'save_dir': r"D:\!DTStack\Savefile",
    'threshold':0.01,
    'operator':OP_DICT,
    'selected': '!f',
    'pre_model_param':model_param,
    'percent':1.0,
}

from common.utils import LogHandler,log,_read_pickle,_save_pickle,create_dir
logHandler = LogHandler()._log


from common.dataset_transfer import OperatorParser
operatorParser = OperatorParser()

PROCESS_NUM = 8

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import catboost as cb


"""
FOR DEBUG ONLY.
"""
np.set_printoptions(threshold=np.inf)
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

class DatasetPool():

    def __init__(self,param):
        self.param = param
        self.name = param['name']
        self.data_dir = param['data_dir']
        self.save_dir = param['save_dir']
        self.operator = param['operator']


        pre_model_param = param['pre_model_param']
        self.pre_model = pre_model_param['model']
        self.pre_model_setting = pre_model_param['model_settings']

        self.selected = param['selected']
        self.percent_ = param['percent']
        self.dataset = None
        self.dataset_dir = {}
        self.dict_LFEtable = {}
        self.dict_LFEtable_config = None
        self.dict_dataset_config = None

        learner_param = param['learner_param']
        self.learner_name = learner_param['name']
        self.learner_if_clean = learner_param['if_clean']
        self.cleaned_range = learner_param['cleaned_range']
        self.train_cache = learner_param['train_cache']
        self.threshold = learner_param['threshold']

        self.rm_cache = param['rm_cache']


        # from common.dataset_transfer import OperatorParser
        # self.operatorParser = OperatorParser()


        self.pool = None
        self.dataset_input = {}
        self.dataset_input_oprtr = {}


    @log(_log = logHandler)
    def run(self,**kwargs):
        self.dataset_preprocessing(**kwargs)
        print (self.dataset_config)
        self.operator_pretraining(**kwargs)

    @log(_log = logHandler)
    def data_forward(self,**kwargs):
        self.dataset_preprocessing(**kwargs)
        print (self.dataset_config)
        self.run_LFE_learner(**kwargs)
        return

    def run_LFE_learner(self):

        for oprtr in self.operator:
            oprtr_dataset_path = os.path.join(self.save_dir, 'dataset', self.learner_name, oprtr, 'data.csv')
            create_dir(oprtr_dataset_path)

            begin_time = time()

            if self.train_cache:
                try:
                    operator_dataset = self.load_from_csv(oprtr_dataset_path)
                except :
                    operator_dataset = self.prepare_operator_dataset(oprtr)
                    self.save_csv_from_df(oprtr_dataset_path, operator_dataset)

            else:
                operator_dataset = self.prepare_operator_dataset(oprtr)
                self.save_csv_from_df( oprtr_dataset_path, operator_dataset)

            end_time = time()
            print('Time Usage for op {} dataset create is : {:.2f}'.format(oprtr, (end_time - begin_time)))
            logHandler.info('Time Usage for op {} dataset create is : {:.2f}'.format(oprtr, (end_time - begin_time)))

        return


    def prepare_operator_dataset(self,oprtr:str)->pd.DataFrame:

        df_lfe_table = copy.deepcopy(self.dict_LFEtable[oprtr])

        # trained on selected dataset.
        df_lfe_table = df_lfe_table.loc[
            df_lfe_table['dataset_name'].isin(self.dataset)]
        df_lfe_table = df_lfe_table.set_index(['dataset_name', 'label', 'feature'])

        print ('@{} Dataset Processing...'.format(oprtr))
        logHandler.info('@{} Dataset Processing...'.format(oprtr))

        if self.learner_if_clean:
            df_lfe_table = self.clean_data(df_lfe_table,range = self.cleaned_range)

        threshed_oprtr_performance = self.threshold_forward(df_lfe_table)
        return self.QSA_forward(oprtr,threshed_oprtr_performance)



    def clean_data(self,data,range = [-0.005/2,0.005/2]):
        return data[(data['performance']<range[0]) | (data['performance']>range[1])]



    def threshold_forward(self,df_lfe_table,autoset=True,fixed_threshold = 50):
        from common.cal_utils import threshold_cut
        # print(df_lfe_table)
        threshold_autoset =  np.percentile(df_lfe_table['performance'].dropna(), fixed_threshold)
        # threshold_autoset = threshold_autoset.astype(np.float32)
        logHandler.info('Chosen threshold: {}'.format(self.threshold)   )
        logHandler.info('{} equal threshlod: {}'.format(fixed_threshold,threshold_autoset) )

        if self.threshold is None:
            if threshold_autoset > 0.01:
                print('threshold down to 0.01.')
                threshold_autoset = 0.01
                __threshold = threshold_autoset
            else:
                print('Changed to {} equal threshold.'.format(fixed_threshold))
                __threshold = threshold_autoset
        else:
            __threshold = self.threshold


        threshed_oprtr_performance = pd.DataFrame(threshold_cut(df_lfe_table['performance'], __threshold).dropna())
        threshed_oprtr_performance = threshed_oprtr_performance.reset_index()

        # threshed_oprtr_performance.drop(labels=['label'],axis=1,inplace=True)
        print("Sampled number: {} ".format(len(threshed_oprtr_performance)))
        print ('pos percent : {}'.format(
            np.array(threshed_oprtr_performance['performance']).sum() / len(threshed_oprtr_performance)
            )
        )

        logHandler.info("Sampled number: {} ".format(len(threshed_oprtr_performance)))
        logHandler.info("Threshlod: {}".format(__threshold))

        return threshed_oprtr_performance



    def QSA_forward(self,oprtr,threshed_oprtr_performance,__range = (-10,10)):
        from common.cal_utils import quantileSkrechArray
        oprtr_array = []
        tmp_df = {}

        for pivoti in threshed_oprtr_performance.index:
            __dataset_name = threshed_oprtr_performance.loc[pivoti]['dataset_name']
            __feature = threshed_oprtr_performance.loc[pivoti]['feature']
            __label = threshed_oprtr_performance.loc[pivoti]['label']
            __class = threshed_oprtr_performance.loc[pivoti]['performance']
            # print(__dataset_name,__feature,__class)
            """
            debug;
            """
            __df_raw_data = copy.deepcopy( self.load_dataset_data(__dataset_name)[0].iloc[:,__feature] )

            if (__dataset_name,__label) not in tmp_df.keys():
                __df_raw_label = copy.deepcopy( self.load_dataset_data(__dataset_name)[0].iloc[:,-1] )
                # print(__label)

                logHandler.info(  __label  )
                logHandler.info( list( __df_raw_label.unique() )  )

                __df_raw_label = __df_raw_label.apply(str)
                __df_raw_label.loc[(__df_raw_label != str(__label) )] = 'F'


                # __df_raw_data.ix[(__df_raw_data.iloc[:,-1] == __label),-1] = 'T'

                #tmp cache
                tmp_df[(__dataset_name,__label)] = __df_raw_label

            else:
                __df_raw_label = tmp_df[(__dataset_name,__label)]

            __df_data = pd.concat([__df_raw_data,__df_raw_label],axis=1)

            #trans for df.

            __df_data.iloc[:,0] = operatorParser.feature_trans(oprtr,__df_data.iloc[:,0])
            # print (__df_data)

            __QSA_data = np.array(
                __df_data.iloc[:,0].groupby(__df_data.iloc[:, -1]).apply(quantileSkrechArray,range=__range)
            ).reshape(-1)

            oprtr_array.append(
                np.append(__QSA_data,__class)
            )
        # print(np.array(oprtr_array))
        return pd.DataFrame(oprtr_array)


    # dataframe [dataset_Name , operater]
    # dataset_Name: [dataset_1vR_label]
    def dataset_preprocessing(self,**kwargs):
        begin_time = time()

        self.dataset = self.load_dataset_name()
        logHandler.info(self.dataset)
        self.dataset_dir = self.load_dataset_dir()

        #load dataset config.
        self.dataset_config = self.load_dataset_config()
        self.update_dataset_config()
        logHandler.info(self.dataset_config)
        #save dataset config.
        self.save_dataset_config()

        # load LFEtable, each operator has a table stocked pre-learned performace on all existing datasets.
        self.operator = self.operator
        for oprtr in self.operator:
            self.dict_LFEtable[oprtr] = self.load_LFE_table(oprtr)

            #rm cache
            if self.rm_cache:
                self.remove_LFE_table(oprtr,self.dataset)

            # load LFEtable for new dataset_config.
            self.update_LFE_table(oprtr)

            # save LFEtable.
            self.save_LFE_table(oprtr)
            # print (self.dict_LFEtable[oprtr])

        end_time = time()
        print('Time Usage for data preprocessing is : {:.2f}'.format((end_time - begin_time)))
        logHandler.info('Time Usage for data preprocessing is : {:.2f}'.format((end_time - begin_time)))


    def operator_pretraining(self,**kwargs):
        for oprtr in self.operator:
            begin_time = time()
            self.training_operator_performance(oprtr)
            end_time = time()
            print('Time Usage for op {} pre-training is : {:.2f}'.format(oprtr, (end_time - begin_time)))
            logHandler.info('Time Usage for op {} pre-training is : {:.2f}'.format(oprtr, (end_time - begin_time)))

    # train twice for oprtr on all dataset given.
    # the first one with origin features and second on oprtred features.


    def training_operator_performance(self,oprtr:str) -> None:

        #load LFE table for oprtr
        df_lfe_table = self.dict_LFEtable[oprtr]
        df_lfe_table = df_lfe_table.set_index(['dataset_name','label','feature'])

        #synchronize dataset_for_training for current LFE table.
        #For DataFrame "oprtr" -> columns = [dataset_name, label, performance]
        # LFE table -> fetch previous progress.
        for __dataset_name in self.dataset:
            __df_raw_data = self.load_dataset_data(__dataset_name)[0].sample(frac = 1)

            # sampling data to speed up traning phase.
            per_num = int(self.percent_ * len(__df_raw_data.index))
            print(per_num)
            __df_raw_data = __df_raw_data.iloc [ 0: per_num, : ]

            #downsample data to avoid too much dimision of features.
            # __df_raw_data[:,0:-1] = self.feature_reduction(__df_raw_data[:,0:-1])

            for __label in df_lfe_table.loc[__dataset_name].index.get_level_values(0).unique():

                # If still nan values remained in LFE table for [__dataset_name,__label].
                FLAG = 1
                for idx in df_lfe_table.loc[__dataset_name,__label].index:
                    if df_lfe_table.loc[__dataset_name,__label].loc[idx].isnull().values[0]:
                        FLAG = 0
                if FLAG:
                    continue

                # for origin feature scored performance.
                start_time = time()

                logHandler.info("{}{}".format('TAG:', str([__dataset_name, __label])))
                score_origin = self.run_training_model(df_raw_data=__df_raw_data,dataset_name = __dataset_name, label = str(__label))
                logHandler.info(  '{}{}'.format('Original scores: ',score_origin)  )

                end_time = time()
                print('Time usage is: %0.2f' % (end_time - start_time))
                logHandler.info('Time usage is: %0.2f' % (end_time - start_time))


                pool = Pool(processes = PROCESS_NUM )
                for __feature in df_lfe_table.loc[__dataset_name,__label].index.get_level_values(0).unique():

                        logHandler.info( "{}{}".format(  'TAG:',str([__dataset_name,__label,__feature]) ) )
                        if  df_lfe_table.loc[__dataset_name,__label,__feature].isnull().values[0]:

                            start_time = time()

                            # for trans features.
                            __df_trans_raw_data = copy.deepcopy(__df_raw_data)
                            __df_trans_raw_data.iloc[:,__feature] = operatorParser.feature_trans(oprtr,__df_raw_data.iloc[:,__feature])
                            # print(__df_trans_raw_data.head(10))

                            score_trans = pool.apply_async( self.run_training_model,
                            (
                            __df_trans_raw_data,
                            __dataset_name,
                            str(__label), )
                            ).get()
                            # score_trans = self.run_training_model(df_raw_data=__df_trans_raw_data,dataset_name = __dataset_name, label = __label)

                            logHandler.info("{}{}".format( 'SUCC SCORED: ', (score_trans - score_origin)  ) )

                            df_lfe_table.loc[__dataset_name,__label,__feature] = (score_trans - score_origin)

                            # update LFE table.
                            self.dict_LFEtable[oprtr] = df_lfe_table.reset_index()
                            self.save_LFE_table(oprtr)

                            end_time = time()
                            print ( 'Time usage is: %0.2f' % (end_time - start_time) )
                            logHandler.info('Time usage is: %0.2f' % (end_time - start_time))

                pool.close()
                pool.join()

        return

        for __dataset in self.dataset:
            raise  NotImplementedError

        #train for each dataset.

        return

    def feature_reduction(self,df_data:pd.DataFrame,n_components=49)->pd.DataFrame:
        from sklearn.decomposition import PCA

        if df_data.shape[1] <= n_components :
            return df_data

        else:
            pca = PCA(n_components=n_components)
            return df_data.fit_transform(df_data)


    def run_training_model(self,df_raw_data,dataset_name,label):

        # df_raw_data = df_raw_data.sample(frac=1)
        df_raw_data_label = copy.deepcopy( (df_raw_data.iloc[:,-1].apply(str) ) )

        #trans to 1vR task.
        df_raw_data_label[df_raw_data_label!=label] = 'non_label'
        labelendr = preprocessing.LabelEncoder()

        data_y = labelendr.fit_transform(df_raw_data_label)
        data_x = df_raw_data.iloc[:,0:-1].values

        logHandler.info(str(data_x.shape))
        logHandler.info(data_x.mean())
        # logHandler.info(str(data_y))

        # # fit into [example , feature]
        # labels = np.array(labels).reshape(len(labels), 1)
        # onehot = preprocessing.OneHotEncoder()
        # onehot_label = onehot.fit_transform(labels)
        # np_data_y = onehot_label.toarray()

        # feed into model.
        model = eval(self.pre_model)
        clf_svc_cv = model(**self.pre_model_setting)
        # clf_svc_cv = GradientBoostingClassifier()

        # cbmodel = cb.CatBoostClassifier(iterations=100,silent=True)
        # cbmodel.fit(data_x,data_y,plot = False,silent = True)
        # scores_clf_cv = cbmodel.score(data_x, data_y)

        """
        for fastbacktest.
        """
        # clf_svc_cv.fit( X=data_x, y=data_y )
        # scores_clf_cv = clf_svc_cv.score( X=data_x, y=data_y)

        # print(  np.sort(clf_svc_cv.feature_importances_)   )
        # print (  np.argsort(clf_svc_cv.feature_importances_)  )
        """
        for cv backtest.
        """

        scores_clf_cv = cross_val_score(clf_svc_cv, data_x, data_y, cv = 5)
        #
        print(scores_clf_cv)
        print("Accuracy: %f (+/- %0.4f)" % (scores_clf_cv.mean(), scores_clf_cv.std() * 2))

        return scores_clf_cv.mean()

    def load_dataset_name(self)->list:
        retdir = []
        tmpdir = os.listdir(self.data_dir)
        selected = self.param['selected'] if 'selected' in self.param.keys() else None
        if selected is None:
            return tmpdir
        else:
            for i in tmpdir:
                if i.startswith(selected):
                    retdir.append(i)
            return retdir

    def load_dataset_dir(self):
        dict_datasetdir = {}
        for name in self.dataset:
            dict_datasetdir[name]=[]
            for root, dirs, files in os.walk(os.path.join(self.data_dir,name)):
                for file in files:
                    __path = os.path.join(root, file)
                    # print(__path)
                    if __path.endswith('mat'):
                        continue

                    if __path.endswith('csv'):
                        dict_datasetdir[name].append(__path)

                    if __path.endswith('txt'):
                        continue
        return dict_datasetdir

    def load_dataset_config(self) -> dict:
        load_path = os.path.join(self.save_dir,self.name,'dataset_config.pickle')
        if os.path.exists(load_path):
            print('Succ load dataset conf.')
            return _read_pickle(load_path)
        else:
            return {}

    def update_dataset_config(self)-> None:
        if self.dataset_config is not None:
            kets = list(self.dataset_config.keys())
        else:
            kets = []
        counter = 0
        for name in self.dataset:
            counter= counter + 1
            if name not in kets:
                begin_time = time()
                self.dataset_config[name] = self.gather_dataset_information(name)
                end_time = time()
                print('Time Usage for {} meta info gathering is : {:.2f}'.format(name,(end_time - begin_time)))
            print ("{} / {} remained.".format ( len(self.dataset) - counter, len(self.dataset) ) )

    def save_dataset_config(self):
        save_path = os.path.join(self.save_dir,self.name,'dataset_config.pickle')
        _save_pickle(self.dataset_config,save_path)

    # dataset_name, dataset_shape, label
    def gather_dataset_information(self,dataset_name) -> dict:
        dataset_info = {}
        dataset_info['name'] = dataset_name
        _, label , shape = self.load_dataset_data(dataset_name)
        dataset_info['label'] = label
        dataset_info['shape'] = shape
        return dataset_info

    #fetch all data through dataset name.
    def load_dataset_data(self,dataset_name:str)-> (pd.DataFrame,list,list):
        if dataset_name in self.dataset_input.keys():
            return self.dataset_input[dataset_name],None,None
        df_data = pd.DataFrame()
        for __dataset_dir in self.dataset_dir[dataset_name]:
            df_data = pd.concat([df_data,self.load_from_csv(__dataset_dir)],axis = 0)

        # For large memory.
        self.dataset_input[dataset_name] = df_data
        # print (df_data)
        label = [ str(lbl) for lbl in df_data.iloc[:,-1].unique()]
        shape = list(df_data.shape)
        return df_data,label,shape


    def remove_LFE_table(self,operator:str,dataset_name)->None:
        df_lfe_table = self.dict_LFEtable[operator]

        df_lfe_table = df_lfe_table[~df_lfe_table['dataset_name'].isin(dataset_name)]
        # print(df_lfe_table)

        self.dict_LFEtable[operator] = df_lfe_table
        return None

    def save_LFE_table(self,operator:str)->None:
        df_lfe_table = self.dict_LFEtable[operator]
        save_path = os.path.join(self.save_dir,self.name,operator,'lfe_table.csv')
        fn = os.path.abspath(save_path)
        create_dir(fn)
        self.save_csv_from_df(fn,df_lfe_table,True)
        return None

    # LFE_table : index: serial INT, columns:['dataset_name','label','performance']
    def load_LFE_table(self,operator:str)->pd.DataFrame :
        load_path = os.path.join(self.save_dir,self.name, operator, 'lfe_table.csv')
        if os.path.exists(load_path):
            print('Succ load LFE table for op {}.'.format(operator))
            data_type = {'label':np.str}
            return pd.read_table(load_path, sep = ',', dtype = data_type)
        else:
            return None

    def update_LFE_table(self,oprtr:str)->None:

        # no LFE currently.
        if self.dict_LFEtable[oprtr] is None:
            # create DataFrame for oprtr.
            df_oprtr = pd.DataFrame(columns=['dataset_name','label','feature','performance'])
            for name in self.dataset:
                for __label in self.dataset_config[name]['label']:
                    for __feature in range(self.dataset_config[name]['shape'][1] - 1):
                        df_oprtr = df_oprtr.append({'dataset_name': name ,
                                                    'label': __label,
                                                    'feature': __feature ,
                                                   'performance':np.nan},ignore_index=True)
            self.dict_LFEtable[oprtr] = df_oprtr

        # NEW dataset for existed oprtr.
        else:
            df_oprtr = self.dict_LFEtable[oprtr]
            df_oprtr_dataset_name = list(df_oprtr['dataset_name'].unique())
            for name in self.dataset:
                if name not in df_oprtr_dataset_name:
                    for __label in self.dataset_config[name]['label']:
                        for __feature in range(self.dataset_config[name]['shape'][1] - 1):
                            df_oprtr = df_oprtr.append({'dataset_name': name ,
                                                        'label': __label,
                                                        'feature': __feature,
                                                        'performance':np.nan}, ignore_index=True)

            self.dict_LFEtable[oprtr] = df_oprtr

    # aborted.

    # def load_LFE_table_config(self)->dict:
    #     load_path = os.path.join(self.save_dir,self.name,'lfe_table_config.json')
    #     if os.path.exist(load_path):
    #         return self.read_json(load_path)
    #     else:
    #         return None

    def save_csv_from_df(self,__path, df: pd.DataFrame,header = None) -> None:
        return df.to_csv(__path, header = header, index = None)

    def load_from_csv(self,__path: str) -> pd.DataFrame:
        df_csv = pd.read_csv(__path,header = None)
        return df_csv

    """
    for hdfs pipeline
    """


if __name__ == '__main__':


    pa = DatasetPool(PARAM_TEST)
    # pa.dataset_preprocessing()
    # print(pa.dict_LFEtable)
    pa.run()
    # print (pa.dict_LFEtable)
    # print (pa.dataset_config)
