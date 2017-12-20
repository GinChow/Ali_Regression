import pickle
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

traincol = pickle.load(open("trainNancol.plk", 'rb'))

testcol = pickle.load(open("testNancol.plk", 'rb'))

df = pd.read_excel("tianchiData/train_data.xlsx")
dftopredict = pd.read_excel("tianchiData/TestA_data.xlsx")

print(df)
print(dftopredict)

df.drop('Y', axis=1, inplace=True)
df = pd.concat([df, dftopredict], axis=0)
print(df.shape)

print(len(traincol))
print(len(testcol))
a = [i for i in traincol if i not in testcol]

for c in a:
    print(df[c].mode().values[0])
    df[c].fillna(value=df[c].mode().values[0], inplace=True)

print(df.loc[:100, :])
print(df.loca[-100:, :])

# print(df['311X28'])
# print(dftopredict['311X28'])
