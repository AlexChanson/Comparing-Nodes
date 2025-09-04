import numpy as np
from sklearn import preprocessing

def load_iris(path="./sample_data/iris.csv"):
    features = []
    nodes = []
    with open(path) as f:
        first_line = f.readline()
        features = first_line.strip().split(',')[:-1]
        for line in f:
            line = line.strip()
            nodes.append(list(map(float, line.split(',')[:-1])))

    return features, np.asarray(nodes)

def normalize(data):
    scaled = preprocessing.MinMaxScaler().fit_transform(data)
    return scaled