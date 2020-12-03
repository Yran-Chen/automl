# python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/11/19 10:28
# @Author  : Yran CHan
# @Site    : 
# @File    : dataset_pool.py
# @Software: PyCharm

import numpy as np
import pandas as pd
import os
import json
import codecs
import pickle
import copy
from time import *

from common.utils import _read_pickle,_save_pickle,create_dir

class DatasetPool():

    def __init__(self,param):
        self.param = param
        self.name = param['name']
        self.data_dir = param['data_dir']
        self.save_dir = param['save_dir']

        self.selected = param['selected']
        self.dataset = None
        self.dataset_dir = {}
        self.dict_LFEtable = {}
        self.dict_LFEtable_config = None
        self.dict_dataset_config = None

        self.dataset_input = {}

    def dataset_preprocessing(self,**kwargs):
        begin_time = time()

        self.dataset = self.load_dataset_name()
        self.dataset_dir = self.load_dataset_dir()

        #load dataset config.
        self.dataset_config = self.load_dataset_config()
        self.update_dataset_config()

        #save dataset config.
        self.save_dataset_config()

        end_time = time()
        print('Time Usage for data preprocessing is : {:.2f}'.format((end_time - begin_time)))

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

        label = [ str(lbl) for lbl in df_data.iloc[:,-1].unique()]
        shape = list(df_data.shape)
        return df_data,label,shape

    def save_csv_from_df(self,__path, df: pd.DataFrame,header = None) -> None:
        return df.to_csv(__path, header = header, index = None)

    def load_from_csv(self,__path: str) -> pd.DataFrame:
        df_csv = pd.read_csv(__path,header = None)
        return df_csv
