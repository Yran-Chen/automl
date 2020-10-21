from common import parser
import pandas as pd
import numpy as np


DICT_R_ARY ={
    1 : ['sqrt', 'log', 'freq', 'zscore','square',
         'sigmoid','cbrt','stdscaler','zero','freq'
        ],

    2 : ['add', 'sub', 'mul', 'div',
        ]
}

class OperatorParser():

    def __init__(self):
        self.parser = parser.Parser()

    def parser_eval(self, expression, factor_dict):
        expr = self.parser.parse(expression)
        expr_parsed = expr.evaluate(factor_dict)
        return expr_parsed

    def feature_trans(self,oprtr:str, df_data:pd.DataFrame)-> pd.DataFrame:
        dict_parse = {
            'data':df_data,
        }
        ary_num = [k for k, v in DICT_R_ARY.items() if oprtr == v or oprtr in v][0]
        if ary_num == 1:
            expr = '{}(data)'.format(oprtr)
            df_trans = self.parser_eval(expr,dict_parse)
        return np.array(df_trans)

if __name__ == '__main__':
    data3 = np.array([0, 1, 3, 4,
                      0, 2, 4, 1,
                      1, 3, 5, 3,
                      1, 4, 6, 5,
                      0, 5, 7, 3]).reshape(5, 4)
    dfp = pd.DataFrame(data3, index=list('abcde'), columns=['four', 'one', 'three', 'two'])

    x = OperatorParser()

    # print(x.feature_trans('stdscaler',dfp))
    # print(x.feature_trans('zscore',dfp))
    # print(dfp)
    tran_dfp = dfp
    print(tran_dfp)
    tran_dfp.loc[:,'two'] = x.feature_trans('freq',dfp.loc[:,'two'])
    print(tran_dfp)