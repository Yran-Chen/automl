# python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/10/19 16:18
# @Author  : Yran CHan
# @Site    : 
# @File    : cal_utils.py
# @Software: PyCharm

import pandas as pd
import numpy as np


# df already been normalized to [-1,1].
def quantileSkrechArray(npary,bins = 200, range=(-10,10))->pd.DataFrame:
    tpary = np.interp(npary,(npary.min(),npary.max()),(range[0], range[1]))
    # print(tpary)
    return pd.DataFrame(np.histogram(tpary,bins=bins,range = range)[0])

if __name__ == '__main__':

    df = pd.DataFrame(data=np.random.randint(-200,200,500)/200)
    print(df)
    print(quantileSkrechArray(df))



def threshold_cut(df, thresh):

    return pd.cut(df,[-1,thresh,1], labels=[0,1])



def quantilize(df, qcut=2, weights='equal', if_opposite=False):
    def quantile_calc(x, q):
        return pd.qcut(x, q=q, labels=False, duplicates='drop')

    print(qcut)
    qindex = df.groupby('date').apply(quantile_calc, qcut)
    lower_ = qindex[qindex == (0.0)].index.tolist()
    upper_ = qindex[qindex == (qcut - 1)].index.tolist()
    ignored_ = qindex[(qindex < (qcut - 1)) & (qindex > (0.0))].index.tolist()
    if (weights == 'equal'):
        df.loc[lower_] = (1.0) * (-1.0 * (-1 + 2 * int(if_opposite)))
        df.loc[upper_] = (-1.0) * (-1.0 * (-1 + 2 * int(if_opposite)))
    elif (weights == 'weighted'):
        df.loc[upper_] = (-1.0) * df.loc[upper_] * (-1.0 * (-1 + 2 * int(if_opposite)))
        df.loc[lower_] = (1.0) * df.loc[lower_] * (-1.0 * (-1 + 2 * int(if_opposite)))
    df.loc[ignored_] = np.NaN
    return df