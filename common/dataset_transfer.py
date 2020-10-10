from common import parser
import pandas as pd
import numpy as np


DICT_R_ARY ={
    1 : ['sqrt', 'log', 'freq', 'zscore','square',
         'sigmoid','cbrt','stdscaler'
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
        return pd.DataFrame(df_trans)

if __name__ == '__main__':
    data3 = np.array([4, 1, 3, 2, 5, 2, 4, 3, 6, 3, 5, 4, 7, 4, 6, 5, 8, 5, 7, 6]).reshape(5, 4)
    dfp = pd.DataFrame(data3, index=list('abcde'), columns=['four', 'one', 'three', 'two'])

    x = OperatorParser()

    print(x.feature_trans('stdscaler',dfp))
    print(x.feature_trans('zscore',dfp))
    print(x.feature_trans('sigmoid',dfp))