# python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/10/10 14:35
# @Author  : Yran CHan
# @Site    : 
# @File    : main_remote.py.py
# @Software: PyCharm

from dataset_for_classifier import DatasetPool
import argparse
__DEBUG = False

parser = argparse.ArgumentParser(description='construct pretrain settings.')

parser.add_argument('--op', type=str, default = None)
parser.add_argument('--selected', type=str, default = None)
parser.add_argument('--percent', type=float, default = None)
parser.add_argument('--name',type=str,default=None)
parser.add_argument('--rm_cache',type=bool,default=None)
args = parser.parse_args()

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

model_param_logreg = {
    'model': 'LogisticRegression',
    "model_settings" : {
    # 'max_iter': 100,
    },
}

OP_DICT = [
    # 'add','sub','mul','div',
    # 'sqrt','log','square','zscore','sigmoid',
    'square',
]

PARAM_TEST = {
    'name': 'test',
    'data_dir': '/data/!workspace/dataset/UCI/classification',
    'save_dir': '/data/!workspace/Savefile',
    'threshold':0.01,
    'operator':OP_DICT,
    'selected': '!f',
    'pre_model_param':model_param_logreg,
    'percent':1.0,
    'rm_cache':False,
}
if __name__ == '__main__':

    if args.op is not None:
        PARAM_TEST['operator'] = args.op.split(',')

    if args.selected is not None:
        PARAM_TEST['selected'] = args.selected

    if args.percent is not None:
        PARAM_TEST['percent'] = args.percent

    if args.name is not None:
        PARAM_TEST['name'] = args.name

    if args.rm_cache is not None:
        PARAM_TEST['rm_cache'] = args.rm_cache

    print(PARAM_TEST)

    pa = DatasetPool(PARAM_TEST)
    pa.run()
