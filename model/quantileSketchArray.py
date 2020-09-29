import numpy as np
import pandas as pd

# df already been normalized to [-1,1].
def quantileSkrechArray(npary,bins = 200, range=(-1,1))->pd.DataFrame:
    return pd.DataFrame(np.histogram(npary,bins=bins,range = range)[0])

if __name__ == '__main__':

    df = pd.DataFrame(data=np.random.randint(-200,200,500)/200)
    print(df)
    print(quantileSkrechArray(df))
