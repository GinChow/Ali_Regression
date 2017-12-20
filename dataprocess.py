import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

def calnans(df):
    thecol = df[df.columns].isnull().any()
    nancol = thecol[thecol.values == True].index
    muchnan = []
    for c in nancol:
        if df[c].isnull().sum() > df.shape[0]//2:
            muchnan.append(c)
    print(df.shape[0])
    print(muchnan)
    pickle.dump(muchnan, open("testNancol.plk", 'wb'))

def fillotherNanColumn(df):
    thecol = df[df.columns].isnull().any()
    nancol = thecol[thecol.values == True].index
    for c in nancol:
        df[c].fillna()

def dropUnique1col(df):
    feats_counts = df.nunique(dropna=True)
    print(feats_counts.shape[0])

    constant_features = feats_counts.loc[feats_counts == 1].index.tolist()
    df.drop(constant_features, axis=1, inplace=True)

df = pd.read_excel("tianchiData/train_data.xlsx")
dftopredict = pd.read_excel("tianchiData/TestA_data.xlsx")

datecol = pickle.load(open("datecol.plk", 'rb'))

# print("df nan")
# print(df[datecol].isnull().describe())
# print(df.isnull().sum().sum())
# print("test nan:")
# print(dftopredict.isnull().describe())
thecol = dftopredict[dftopredict.columns].isnull().any()
nancol = thecol[thecol.values==True].index
print(nancol)
rows = dftopredict[dftopredict[datecol].isnull().T.any().T]
newdf = dftopredict.drop(rows.index, axis=0)

calnans(dftopredict)



# for c in nancol:
#     dftopredict[c] = dftopredict[c].fillna(value=newdf[c].median())
#
# print(dftopredict.loc[rows.index, nancol])
# print("origin:")
# print(dftopredict[nancol].isnull().any())
#
# for c in nancol:
#     newdf[c] = pd.to_datetime(newdf[c], format='%Y%m%d', exact=False)
#     print(newdf[c])
#
#
# for c in nancol:
#     newdf[c].plot()
#     plt.show()

# print("df nan:")
# print(df.isnull())
#
# print("test nan:")
# print(dftopredict.isnull())
#
# nan_rows = df[df.isnull().T.any().T]
# print("nan_rows")
# print(nan_rows)
#
# nanpredict_rows = dftopredict[dftopredict.isnull().T.any().T]
# print("nanpredict_rows")
# print(nanpredict_rows)
# print(dftopredict)
# dfcopy = df.copy()
#
# y = dfcopy.loc[:, 'Y']
# dfcopy = dfcopy.drop('Y', axis=1)   # slice the label Y
# # dftopredict = dftopredict.fillna(-999)
# # dfcopy = dfcopy.fillna(-999)
#
# dfcopy = pd.concat([dfcopy, dftopredict], axis=0)
# print(dfcopy.iloc[-100:, :])
# print(dfcopy.shape)
#
# # load datetype columns
# datecol = pickle.load(open("datecol.plk", 'rb'))
#
# dfcopy = dfcopy.drop('ID', axis=1)