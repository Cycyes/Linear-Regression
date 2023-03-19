import copy
import numpy as np
import random

def train_test_split(x, x_scaled, x_pca2, x_pca4, x_pca6, x_pca8, x_pca10, y, test_radio):
    cnt = int(len(y) * test_radio)
    s = [i for i in range(0, len(y))]
    random.shuffle(s)
    return np.array(x)[s[:len(y) - cnt]], np.array(x)[s[len(y) - cnt:len(y)]], np.array(x_scaled)[s[:len(y) - cnt]], np.array(x_scaled)[s[len(y) - cnt:len(y)]], np.array(x_pca2)[s[:len(y) - cnt]], np.array(x_pca2)[s[len(y) - cnt:len(y)]], np.array(x_pca4)[s[:len(y) - cnt]], np.array(x_pca4)[s[len(y) - cnt:len(y)]], np.array(x_pca6)[s[:len(y) - cnt]], np.array(x_pca6)[s[len(y) - cnt:len(y)]], np.array(x_pca8)[s[:len(y) - cnt]], np.array(x_pca8)[s[len(y) - cnt:len(y)]], np.array(x_pca10)[s[:len(y) - cnt]], np.array(x_pca10)[s[len(y) - cnt:len(y)]], np.array(y)[s[:len(y) - cnt]], np.array(y)[s[len(y) - cnt:len(y)]]

def PCA(x, k=10):
    x_mean = np.mean(x, axis=0)
    X = x - x_mean
    cov = np.cov(X, rowvar=0)
    eVals, eVects = np.linalg.eig(np.mat(cov))
    eValInd = np.argsort(eVals)
    eValInd = eValInd[: -(k + 1): -1]
    E = eVects[:, eValInd]
    lowDMatrix = X * E
    return lowDMatrix

def PR(x, k=3):
    ret = []
    for i in x:
        e = copy.deepcopy(i)
        for j in range(2, k + 1):
            e += (np.array(i) ** j).tolist()
        ret.append(e)
    return ret

def FeatureScaling(x):
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    return (x - x_mean) / x_std


