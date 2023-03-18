import copy
import numpy as np

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
    num_item, num_feature = np.shape(x)
    ret = []
    for i in x:
        e = copy.deepcopy(i)
        for j in range(2, k + 1):
            e += (np.array(i) ** j).tolist()
        ret.append(e)
    return ret