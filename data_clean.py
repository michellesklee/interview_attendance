import numpy as np
import pandas as pd
from sklearn import preprocessing

df = pd.read_csv('Interview.csv')
df = df.drop(['Date of Interview', 'Name(Cand ID)'], axis=1)

def one_hot_encoding(df):
    le = preprocessing.LabelEncoder()
    df_int = df.astype(str).apply(le.fit_transform) #transforms strings to integers
    enc = preprocessing.OneHotEncoder()
    enc.fit(df_int)
    df_encode = enc.transform(df_int).toarray() #returns array of one hot encodes
    return df_encode

print(one_hot_encoding(df))
