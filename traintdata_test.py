import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn import preprocessing
from sklearn.svm import SVR
import tensorflow as tf
from tflearn import data_utils
import pickle

valid_size = 100


def filldate(dftopredict, datecol):
    thecol = dftopredict[datecol].isnull().any()
    nancol = thecol[thecol.values == True].index
    rows = dftopredict[dftopredict[datecol].isnull().T.any().T]
    newdf = dftopredict.drop(rows.index, axis=0)

    for c in nancol:
        dftopredict[c] = dftopredict[c].fillna(value=newdf[c].median())

    return dftopredict


def droptoomuchNanCol(df):
    traincol = pickle.load(open("trainNancol.plk", 'rb'))
    testcol = pickle.load(open("testNancol.plk", 'rb'))
    cross = [i for i in traincol if i in testcol]
    return df.drop(cross, axis=1)


def fillrestnancol(df):
    traincol = pickle.load(open("trainNancol.plk", 'rb'))
    testcol = pickle.load(open("testNancol.plk", 'rb'))
    cross = [i for i in traincol if i not in testcol]
    for c in cross:
        df[c].fillna(value=df[c].mode().values[0], inplace=True)
    return df


def dropuselesscol(df):
    # drop useless columns
    feats_counts = df.nunique(dropna=True)

    constant_features = feats_counts.loc[feats_counts == 1].index.tolist()
    df.drop(constant_features, axis=1, inplace=True)
    return df


def dropRepeatCol(df):
    # drop repeat columns
    dfencode = pd.DataFrame(index=df.index)

    obj_cols = []
    for c in df.columns[df.dtypes == 'object']:
        dfencode[c] = df[c].factorize()[0]
        obj_cols.append(c)

    # find repeat columns
    dup_cols = {}
    for i, c1 in enumerate(df.columns[df.dtypes == 'object']):
        for c2 in dfencode.columns[i + 1:]:
            if c2 not in dup_cols and np.all(dfencode[c1] == dfencode[c2]):
                dup_cols[c2] = c1

    df.drop(dup_cols.keys(), axis=1, inplace=True)
    return df, obj_cols, dup_cols


def one_hot(df, obj_cols, dup_cols):
    need_dummycol = [i for i in obj_cols if i not in dup_cols.keys()]
    print("need dummy cols:", need_dummycol)
    dummyed_df = pd.get_dummies(df[need_dummycol])
    df.drop(need_dummycol, axis=1, inplace=True)

    df = pd.concat([df, dummyed_df], axis=1)
    return df


def fillotherNanColumn(df):
    thecol = df[df.columns].isnull().any()
    nancol = thecol[thecol.values == True].index
    for c in nancol:
        df[c].fillna(value=df[c].mode().values[0], inplace=True)
    return df


# load raw data
df = pd.read_excel("tianchiData/train_data.xlsx")
dftopredict = pd.read_excel("tianchiData/TestA_data.xlsx")

# load datetype columns
datecol = pickle.load(open("datecol.plk", 'rb'))

dfcopy = df.copy()
y = dfcopy.loc[:, 'Y']

# dftopredict = filldate(dftopredict, datecol)   # fill nan date data



# dfcopy = dfcopy.fillna(-999)        # fill nan value  ####### TODO can fill with column median
# dftopredict = dftopredict.fillna(-999)

# dfcopy = pd.concat([dfcopy, dftopredict], axis=0)     # concat train and predict data

dfcopy = dfcopy.drop('Y', axis=1)   # slice labels
dfcopy = dfcopy.drop('ID', axis=1)


# dfcopy = dfcopy.drop('311X28', axis=1)

# convert datetime to days baseline 2017.1.1
initdate = pd.datetime(year=2017, month=1, day=1)
for c in datecol:
    dfcopy[c] = pd.to_datetime(dfcopy[c], format="%Y%m%d", exact=False)
    dfcopy[c] = (dfcopy[c] - initdate).dt.days

# drop too much nan columns
dfcopy = droptoomuchNanCol(dfcopy)

# fill rest nan cols with mode  NOW THERE IS NO NAN VALUE!!!!
dfcopy = fillrestnancol(dfcopy)
dfcopy = fillotherNanColumn(dfcopy)

print("Nans:", dfcopy.isnull().sum().sum())

# drop useless columns
dfcopy = dropuselesscol(dfcopy)

# dfcopy = dfcopy.fillna(-999)    # fill nan value
# dftopredict = dftopredict.fillna(-999)

# drop repeat columns
dfcopy, obj_cols, dup_cols = dropRepeatCol(dfcopy)

# one hot
dfcopy = one_hot(dfcopy, obj_cols, dup_cols)

# next do data preprocess
X = np.array(dfcopy)
Y = np.array(y)

scalar = preprocessing.StandardScaler().fit(X)  # zero-score normalize
X_scaled = scalar.transform(X)

print(X_scaled)
print(X_scaled.shape)

# divide into train and valid set
X_scaled, y = data_utils.shuffle(X_scaled, y)

train_x = X_scaled[:-valid_size, :]
train_y = y[:-valid_size]
# train_y = y

valid_x = X_scaled[-valid_size:, :]
valid_y = y[-valid_size:]

print(train_x.shape, train_y.shape)

# print(valid_x.shape, valid_y.shape)

svr_rbf = SVR(kernel='rbf', degree=3)

y_rbf = svr_rbf.fit(train_x, train_y).predict(valid_x)

# pickle.dump(y_rbf, open("testA_res.plk", 'wb'))


print("predict:")
print(y_rbf)

print("target:")
print(valid_y)

print("subtract:")
print(y_rbf - valid_y)

print("MSE:")
print(np.mean((y_rbf - valid_y)**2))















