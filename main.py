from dataset_for_classifier import DatasetPool
import argparse

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
    },
}

model_param_logreg = {
    'model': 'LogisticRegression',
    "model_settings" : {
    # 'max_iter': 100,
    },
}

OP_DICT = ['zscore', 'cbrt', 'sigmoid', 'stdscaler','freq']
OP_DICT_TEST = ['zscore']

DEBUG_learner_param = {
    'name':'test_1112',
    'train_cache':False,
    'if_clean':False,
    'threshold':None,
}

TEST_learner_param = {
    'name':'test_beta',
    'train_cache':False,
    'if_clean':True,
    'threshold': 0.002/3,
    'cleaned_range':[-0.01/6,0.01/6]
}

baseline_learner_param = {
    'name':'baseline',
    'train_cache':False,
    'if_clean':False,
    'threshold':0.01,
}

PARAM_DEBUG = {
    'name':'dataset_test_!f_00408',
    'data_dir': r'D:\!DTStack\Dataset\UCI_\ml\machine-learning-databases',
    'save_dir': r"D:\!DTStack\Savefile",
    'operator':OP_DICT_TEST,
    'selected': '!f_00408',
    'pre_model_param':model_param_logreg,
    'percent':1.0,
    'rm_cache': False,
    'learner_param':DEBUG_learner_param,
}

PARAM_BETA = {
    'name':'lr_beta',
    'data_dir': r'D:\!DTStack\Dataset\UCI_\ml\machine-learning-databases',
    'save_dir': r"D:\!DTStack\Savefile_remote",
    'operator':['stdscaler','zscore','sigmoid','cbrt'],
    'selected': '!f',
    'pre_model_param':model_param_logreg,
    'percent':1.0,
    'rm_cache': False,
    'learner_param':{
                    'name':'test_beta',
                    'train_cache':False,
                    'if_clean':True,
                    'threshold': 0.002/3,
                    'cleaned_range':[-0.01/6,0.01/6]
                    },
}

PARAM_EVAL = {
    'name':'lr_beta_eval',
    'data_dir': r'D:\!DTStack\Dataset\UCI_\ml\machine-learning-databases',
    'save_dir': r"D:\!DTStack\Savefile_remote",
    'operator':OP_DICT,
    'selected': '!eval',
    'pre_model_param':model_param_logreg,
    'percent':1.0,
    'rm_cache': False,
    'learner_param':{
                    'name':'test_beta_eval',
                    'train_cache':False,
                    'if_clean':False,
                    'threshold': 0.02/3,
                    'cleaned_range':[-0.01/3,0.01/3]
                    },
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

    # pa = DatasetPool(PARAM_DEBUG)
    # pa.run()

    pa = DatasetPool(PARAM_EVAL)
    pa.run()
    pa.data_forward()