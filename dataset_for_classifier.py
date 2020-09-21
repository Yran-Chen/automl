import numpy as np
import pandas as pd
import os
import json
import codecs
import pickle
from time import *

from sklearn import preprocessing
import copy
from sklearn.model_selection import cross_val_score
from multiprocessing import Process, Pool

model_param = {
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
}

from common.utils import LogHandler, log, logTime
logHandler = LogHandler()._log


from common.dataset_transfer import OperatorParser
operatorParser = OperatorParser()


class DatasetPool():

    def __init__(self,param):
        self.param = param
        self.name = param['name']
        self.data_dir = param['data_dir']
        self.save_dir = param['save_dir']
        self.operator = param['operator']
        self.threshold = param['threshold']
        self.pre_model_param = param['pre_model_param']
        self.selected = param['selected']
        self.dataset = None
        self.dataset_dir = {}
        self.dict_LFEtable = {}
        self.dict_LFEtable_config = None
        self.dict_dataset_config = None

        # from common.dataset_transfer import OperatorParser
        # self.operatorParser = OperatorParser()


        self.pool = None

        self.dataset_input = {}

    @log(_log = logHandler)
    def run(self,**kwargs):
        self.dataset_preprocessing(**kwargs)
        self.operator_pretraining(**kwargs)


    # dataframe [dataset_Name , operater]
    # dataset_Name: [dataset_1vR_label]
    def dataset_preprocessing(self,**kwargs):
        begin_time = time()

        self.dataset = self.load_dataset_name()
        print(self.dataset)
        self.dataset_dir = self.load_dataset_dir()

        #load dataset config.
        self.dataset_config = self.load_dataset_config()
        self.update_dataset_config()
        #save dataset config.
        self.save_dataset_config()

        # load LFEtable, each operator has a table stocked pre-learned performace on all existing datasets.
        self.operator = self.operator
        for oprtr in self.operator:
            # load LFEtable for new dataset_config.
            self.dict_LFEtable[oprtr] = self.load_LFE_table(oprtr)
            self.update_LFE_table(oprtr)
            # save LFEtable.
            self.save_LFE_table(oprtr)

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
        for __dataset_name in df_lfe_table.index.levels[0]:
            __df_raw_data = self.load_dataset_data(__dataset_name)[0]

            #downsample data to avoid too much dimision of features.
            # __df_raw_data[:,0:-1] = self.feature_reduction(__df_raw_data[:,0:-1])

            for __label in df_lfe_table.loc[__dataset_name].index.get_level_values(0).unique():
                # for origin feature scored performance.
                # print(__df_raw_data)
                score_origin = self.run_training_model(df_raw_data=__df_raw_data,dataset_name = __dataset_name, label = __label)
                logHandler.info(  '{}{}'.format('Original scores: ',score_origin)  )


                pool = Pool(processes=4)
                for __feature in df_lfe_table.loc[__dataset_name,__label].index.get_level_values(0).unique():

                        logHandler.info( "{}{}".format(  'TAG:',str([__dataset_name,__label,__feature]) ) )
                        if  df_lfe_table.loc[__dataset_name,__label,__feature].isnull().values[0]:

                            start_time = time()

                            # for trans features.
                            __df_trans_raw_data = __df_raw_data
                            __df_trans_raw_data.iloc[:,__feature] = operatorParser.feature_trans(oprtr,__df_raw_data.iloc[:,__feature])

                            score_trans = pool.apply_async( self.run_training_model, ( __df_trans_raw_data, __dataset_name,__label, ) ).get()
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

    # @logTime(_log=logHandler)
    def run_training_model(self,df_raw_data,dataset_name,label,
                           model = 'GradientBoostingClassifierV2Ai',max_num = 10000):

        from sklearn.ensemble import GradientBoostingClassifier

        df_raw_data = df_raw_data.sample(frac=1)
        df_raw_data_label = copy.deepcopy( (df_raw_data.iloc[:,-1].apply(str)) )

        #trans to 1vR task.
        df_raw_data_label[df_raw_data_label!=label] = 'non_label'

        labelendr = preprocessing.LabelEncoder()
        if max_num  > 0 :
            data_y = labelendr.fit_transform(df_raw_data_label)[0:max_num]
            data_x = df_raw_data.iloc[:,0:-1].values[0:max_num,:]
        else:
            data_y = labelendr.fit_transform(df_raw_data_label)
            data_x = df_raw_data.iloc[:, 0:-1].as_matrix()

        logHandler.info(str(data_x.shape))
        # logHandler.info(str(data_y))

        # # fit into [example , feature]
        # labels = np.array(labels).reshape(len(labels), 1)
        # onehot = preprocessing.OneHotEncoder()
        # onehot_label = onehot.fit_transform(labels)
        # np_data_y = onehot_label.toarray()

        # feed into model.
        clf_svc_cv = GradientBoostingClassifier(**self.pre_model_param['model_settings'])
        scores_clf_cv = cross_val_score(clf_svc_cv, data_x, data_y, cv = 5)
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
            return DatasetPool._read_pickle(load_path)
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
        DatasetPool._save_pickle(self.dataset_config,save_path)

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

        label = [ str(lbl) for lbl in df_data.iloc[:,-1].unique()]
        shape = list(df_data.shape)
        return df_data,label,shape


    def save_LFE_table(self,operator:str)->None:
        df_lfe_table = self.dict_LFEtable[operator]
        save_path = os.path.join(self.save_dir,self.name,operator,'lfe_table.csv')
        fn = os.path.abspath(save_path)
        DatasetPool.create_dir(fn)
        self.save_csv_from_df(fn,df_lfe_table,True)
        return None

    # LFE_table : index: serial INT, columns:['dataset_name','label','performance']
    def load_LFE_table(self,operator:str)->pd.DataFrame :
        load_path = os.path.join(self.save_dir,self.name, operator, 'lfe_table.csv')
        if os.path.exists(load_path):
            print('Succ load LFE table for op {}.'.format(operator))
            return pd.read_csv(load_path)
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
                        df_oprtr = df_oprtr.append({'dataset_name': name ,'label': __label, 'performance':np.nan},ignore_index=True)
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

    @staticmethod
    def _save_pickle(file, file_dir):
        fn = os.path.abspath(file_dir)
        DatasetPool.create_dir(fn)
        with open(fn, 'wb') as f:
            pickle.dump(file, f)
        f.close()

    @staticmethod
    def _read_pickle(fp):
        content = dict()
        try:
            with open(fp, 'rb') as f:
                content = pickle.load(f)
        except IOError as e:
            if e.errno not in (errno.ENOENT, errno.EISDIR, errno.EINVAL):
                raise
        return content

    @staticmethod
    def _read_json(fp):
        content = dict()
        try:
            with codecs.open(fp, 'r', encoding='utf-8') as f:
                content = json.load(f)
        except IOError as e:
            if e.errno not in (errno.ENOENT, errno.EISDIR, errno.EINVAL):
                raise
        return content

    @staticmethod
    def _save_json(serializable, file_dir):
        fn = os.path.abspath(file_dir)
        DatasetPool.create_dir(fn)
        with codecs.open(fn, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, separators=(',\n', ': '))

    @staticmethod
    def create_dir(file_dir):
        if not os.path.exists(os.path.dirname(file_dir)):
            try:
                os.makedirs(os.path.dirname(file_dir))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise EnvironmentError
                else:
                    print("???")

    """
    for hdfs pipeline
    """


if __name__ == '__main__':


    pa = DatasetPool(PARAM_TEST)
    # pa.dataset_preprocessing()
    pa.run()
    # print (pa.dict_LFEtable)
    # print (pa.dataset_config)