# -*- coding: utf-8 -*-
# @Date    : 9-8-2020
# @Author  : Yiran Chen
# @Email   : Is.Yiran.Chen@gmail.com

import  numpy as np
import  pandas as pd
import  os


def rename_for_dataset_txt(fpath):
    df_download_url = pd.read_table(fpath, sep='\n', header=None)
    for i in df_download_url.index:
        _url = df_download_url.loc[i].values[0]
        insert_pos = _url.rindex('/')
        p = list(_url)
        p.insert(insert_pos+1,'{}_'.format(i))
        df_download_url.loc[i] = "".join('%s' %id for id in p)
    return df_download_url

    return download_url

def data_reformat(fpath):
    df = pd.read_table(fpath,sep=',',header=None)
    return df

def create_dir_file_for_dataset(file_dir):
    dataset_list = []
    save_dir = "D:\\!DTStack\\Dataset\\uci_classification.txt"
    with open(save_dir, 'w+') as f:
        for root, dirs, files in os.walk(file_dir):
            for file in files:
                __path = os.path.join(root, file)
                print(__path)
                f.write(__path)
                f.write('\n')
                # dataset_list.append(__path)
                # if os.path.splitext(file)[0] == 'bayes':
                #     bayes_list.append(os.path.join(root, file))
                # elif os.path.splitext(file)[0] == 'svm':
                #     svm_list.append(os.path.join(root, file))
                # elif os.path.splitext(file)[0] == 'xgboost':
                #     xgboost_list.append(os.path.join(root, file))
    f.close()
    return

if __name__ == '__main__':
    # df_res = rename_for_dataset_tzt('D:\!DTStack\dataset_uci_classcification.txt')
    # df_res.to_csv('dataset_rename_ordered.csv')
    # print(data_reformat("D:\\!DTStack\Dataset\\UCI_\\ml\\machine-learning-databases\\covtype\\covtype.data"))
    create_dir_file_for_dataset('D:\\!DTStack\\Dataset\\UCI_\\ml\\machine-learning-databases')
    df = pd.DataFrame()
    df.to_csv()