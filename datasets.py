import numpy as np
from sklearn import preprocessing


def load_icij(path="sample_data/Officer_indicators_processed_nonulls.csv"):
    # TODO
    return True

def load_iris(path="./sample_data/iris.csv"):
    features = []
    nodes = []
    with open(path) as f:
        first_line = f.readline()
        features = first_line.strip().split(',')[:-1]
        for line in f:
            line = line.strip()
            nodes.append(list(map(float, line.split(',')[:-1])))
    print("LOADED IRIS |D|=", len(features), " n=", len(nodes))
    return features, np.asarray(nodes)

def load_directors(path="./sample_data/Director_indicators_processed_nonulls.csv"):
    features = []
    nodes = []
    with open(path) as f:
        first_line = f.readline()
        features = first_line.strip().split(',')[1:]
        for line in f:
            line = line.strip()
            nodes.append(list(map(lambda x: float('nan') if x == '' else float(x), line.split(',')[1:])))
    print("LOADED Directors |D|=", len(features), " n=", len(nodes))
    return features, np.asarray(nodes)


def load_movies(path="./sample_data/Movie_indicators_processed_nonulls.csv"):
    features = []
    nodes = []
    with open(path) as f:
        first_line = f.readline()
        features = first_line.strip().split(',')[1:]
        for line in f:
            line = line.strip()
            nodes.append(list(map(lambda x: float('nan') if x == '' else float(x), line.split(',')[1:])))
    print("LOADED Movies |D|=", len(features), " n=", len(nodes))
    return features, np.asarray(nodes)

def normalize(data):
    scaled = preprocessing.MinMaxScaler().fit_transform(data)
    return scaled

def load_airports(path="./sample_data/Airport_indicators_processed_nonulls.csv"):
    features = []
    nodes = []
    with open(path) as f:
        first_line = f.readline()
        features = first_line.strip().split(',')[1:]
        for line in f:
            line = line.strip()
            nodes.append(list(map(lambda x : float('nan') if x == '' else float(x) , line.split(',')[1:])))
    print("LOADED AIRPORTS |D|=", len(features), " n=", len(nodes))
    return features, np.asarray(nodes)
