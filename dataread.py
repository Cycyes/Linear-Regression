import os
import pandas as pd
import numpy as np

def medical_data():
    df_medical = pd.read_csv(os.path.join(os.path.dirname(__file__), 'datasets', 'Medical Cost Personal Datasets.csv'))
    #数值化
    df_medical['sex'] = df_medical['sex'].replace({'female': 0, 'male': 1})
    df_medical['smoker'] = df_medical['smoker'].replace({'yes': 1, 'no': 0})
    df_medical = df_medical.drop(['region'], axis=1)
    y = []
    x = []
    for index, row in df_medical.iterrows():
        #x_medical = 
        t = [i for i in row]
        y.append(t.pop())
        x.append(t)
    return x,y

def test_data_1():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'datasets', 'dump', 'test.csv'))
    y = []
    x = []
    for index, row in df.iterrows():
        t = [i for i in row]
        y.append(t.pop())
        x.append(t)
    return x, y

def train_data_1():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'datasets', 'dump', 'train.csv'))
    y = []
    x = []
    for index, row in df.iterrows():
        t = [i for i in row]
        y.append(t.pop())
        x.append(t)
    return x, y

def get_data_2():
    tmp = []
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'datasets', 'CarPrice_Assignment.csv'))
    col = [i for i in df.columns]
    for i in col:
        if isinstance(df[i][0], str):
            word_arr = []
            for word in df[i]:
                if word not in word_arr:
                    word_arr.append(word)
            for c in word_arr:
                tt = []
                for j in df[i]:
                    if j == c:
                        tt.append(1)
                    else:
                        tt.append(0)
                tmp.append(tt)
        else:
            tmp.append(list(df[i]))
    x = []
    y = tmp.pop()
    x = [list(i) for i in zip(*tmp)]
    return x, y
# print(train_data_1())

# housing

def housing_train_data():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'datasets', 'housing', 'train_dataset.csv'))
    y = []
    x = []
    for index, row in df.iterrows():
        t = [i for i in row]
        y.append(t.pop())
        x.append(t)
    return x, y

def housing_test_data():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'datasets', 'housing', 'test_dataset.csv'))
    y = []
    x = []
    for index, row in df.iterrows():
        t = [i for i in row]
        y.append(t.pop())
        x.append(t)
    return x, y
