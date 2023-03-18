import os
import pandas as pd

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

# print(train_data_1())