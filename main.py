from dataset_for_classifier import DatasetPool

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

if __name__ == '__main__':

    pa = DatasetPool(PARAM_TEST)
    pa.run()