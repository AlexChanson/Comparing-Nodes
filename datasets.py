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

def load_actors(path="./sample_data/Actor_indicators_processed_nonulls.csv"):
    features = []
    nodes = []
    with open(path) as f:
        first_line = f.readline()
        features = first_line.strip().split(',')[1:]
        for line in f:
            line = line.strip()
            nodes.append(list(map(lambda x: float('nan') if x == '' else float(x), line.split(',')[1:])))
    print("LOADED Actors |D|=", len(features), " n=", len(nodes))
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

def load_city(path="./sample_data/City_indicators_processed_nonulls.csv"):
    features = []
    nodes = []
    with open(path) as f:
        first_line = f.readline()
        features = first_line.strip().split(',')[1:]
        for line in f:
            line = line.strip()
            nodes.append(list(map(lambda x : float('nan') if x == '' else float(x) , line.split(',')[1:])))
    print("LOADED CITY |D|=", len(features), " n=", len(nodes))
    return features, np.asarray(nodes)

def load_country(path="./sample_data/Country_indicators_processed_nonulls.csv"):
    features = []
    nodes = []
    with open(path) as f:
        first_line = f.readline()
        features = first_line.strip().split(',')[1:]
        for line in f:
            line = line.strip()
            nodes.append(list(map(lambda x : float('nan') if x == '' else float(x) , line.split(',')[1:])))
    print("LOADED Country |D|=", len(features), " n=", len(nodes))
    return features, np.asarray(nodes)

def load_entity(path="./sample_data/Entity_indicators_processed_nonulls.csv"):
    features = []
    nodes = []
    with open(path) as f:
        first_line = f.readline()
        features = first_line.strip().split(',')[1:]
        for line in f:
            line = line.strip()
            nodes.append(list(map(lambda x : float('nan') if x == '' else float(x) , line.split(',')[1:])))
    print("LOADED Entity |D|=", len(features), " n=", len(nodes))
    return features, np.asarray(nodes)

def load_intermediary(path="./sample_data/Intermediary_indicators_processed_nonulls.csv"):
    features = []
    nodes = []
    with open(path) as f:
        first_line = f.readline()
        features = first_line.strip().split(',')[1:]
        for line in f:
            line = line.strip()
            nodes.append(list(map(lambda x : float('nan') if x == '' else float(x) , line.split(',')[1:])))
    print("LOADED Intermediary |D|=", len(features), " n=", len(nodes))
    return features, np.asarray(nodes)


def load_officer(path="./sample_data/Officer_indicators_processed_nonulls.csv"):
    features = []
    nodes = []
    with open(path) as f:
        first_line = f.readline()
        features = first_line.strip().split(',')[1:]
        for line in f:
            line = line.strip()
            nodes.append(list(map(lambda x : float('nan') if x == '' else float(x) , line.split(',')[1:])))
    print("LOADED Officer |D|=", len(features), " n=", len(nodes))
    return features, np.asarray(nodes)


def load_custom(path, delimiter):
    features = []
    nodes = []
    with open(path) as f:
        first_line = f.readline()
        features = first_line.strip().split(delimiter)
        for line in f:
            line = line.strip()
            nodes.append(list(map(lambda x : float('nan') if x == '' else float(x) , line.split(delimiter))))
    print("LOADED AIRPORTS |D|=", len(features), " n=", len(nodes))
    return features, np.asarray(nodes)
