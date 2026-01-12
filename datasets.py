import math

import numpy as np
from sklearn import preprocessing


def load_icij(path="sample_data/Officer_indicators_processed_nonulls.csv") -> bool:
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
    all = []
    beforeValidation = []
    with open(path) as f:
        first_line = f.readline()
        features = first_line.strip().split(',')[1:]
        for line in f:
            line = line.strip()
            nodes.append([float('nan') if x == '' else float(x) for x in line.split(',')[1:]])
            all.append([float('nan') if x == '' else float(x) for x in line.split(',')])
    print("LOADED Directors |D|=", len(features), " n=", len(nodes))
    #return features, np.asarray(nodes)
    with open("./sample_data/Director_beforeValidation.csv") as f:
        first_line = f.readline()
        for line in f:
            line = line.strip()
            beforeValidation.append([float('nan') if x == '' else float(x) for x in line.split(',')])
    return beforeValidation, all, features, np.asarray(nodes)


def load_actors(path="./sample_data/Actor_indicators_processed_nonulls.csv"):
    features = []
    nodes = []
    with open(path) as f:
        first_line = f.readline()
        features = first_line.strip().split(',')[1:]
        for line in f:
            line = line.strip()
            nodes.append([float('nan') if x == '' else float(x) for x in line.split(',')[1:]])
    print("LOADED Actors |D|=", len(features), " n=", len(nodes))
    return features, np.asarray(nodes)


def load_movies(path="./sample_data/Movie_indicators_processed_nonulls.csv"):
    features = []
    nodes = []
    all = []
    beforeValidation = []
    with open(path) as f:
        first_line = f.readline()
        features = first_line.strip().split(',')[1:]
        for line in f:
            line = line.strip()
            nodes.append([float('nan') if x == '' else float(x) for x in line.split(',')[1:]])
            all.append([float('nan') if x == '' else float(x) for x in line.split(',')])
    print("LOADED Movies |D|=", len(features), " n=", len(nodes))
    #return features, np.asarray(nodes)
    with open("./sample_data/Movie_beforeValidation.csv") as f:
        first_line = f.readline()
        for line in f:
            line = line.strip()
            beforeValidation.append([float('nan') if x == '' else float(x) for x in line.split(',')])
    return beforeValidation, all, features, np.asarray(nodes)


def normalize(data):
    return preprocessing.MinMaxScaler().fit_transform(data)


def load_airports(path="./sample_data/Airport_indicators_processed_nonulls.csv"):
    features = []
    nodes = []
    with open(path) as f:
        first_line = f.readline()
        features = first_line.strip().split(',')[1:]
        for line in f:
            line = line.strip()
            nodes.append([float('nan') if x == '' else float(x) for x in line.split(',')[1:]])
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
            nodes.append([float('nan') if x == '' else float(x) for x in line.split(',')[1:]])
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
            nodes.append([float('nan') if x == '' else float(x) for x in line.split(',')[1:]])
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
            nodes.append([float('nan') if x == '' else float(x) for x in line.split(',')[1:]])
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
            nodes.append([float('nan') if x == '' else float(x) for x in line.split(',')[1:]])
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
            nodes.append([float('nan') if x == '' else float(x) for x in line.split(',')[1:]])
    print("LOADED Officer |D|=", len(features), " n=", len(nodes))
    return features, np.asarray(nodes)


def load_custom_OLD(path, delimiter):
    features = []
    nodes = []
    with open(path) as f:
        first_line = f.readline()
        features = first_line.strip().split(delimiter)
        for line in f:
            line = line.strip()
            nodes.append([float('nan') if x == '' else float(x) for x in line.split(delimiter)])
    #TODO drop columns with Nans
    print("LOADED DATASET ", path," |D|=", len(features), " n=", len(nodes))
    return features, np.asarray(nodes)


def load_custom(path, delimiter):
    features = []
    nodes = []
    with open(path) as f:
        first_line = f.readline()
        features = first_line.strip().split(delimiter)

        for line in f:
            values = []
            for x in line.strip().split(delimiter):
                if x == "":
                    values.append(float('nan'))
                else:
                    try:
                        values.append(float(x))
                    except ValueError:
                        values.append(float('nan'))
            nodes.append(values)

    # --- drop columns containing at least one NaN ---
    cols_to_keep = [
        i for i in range(len(nodes[0]))
        if all(not math.isnan(row[i]) for row in nodes)
    ]

    nodes = [
        tuple(row[i] for i in cols_to_keep)
        for row in nodes
    ]

    features = [features[i] for i in cols_to_keep]
    print("LOADED DATASET ", path, " |D|=", len(features), " n=", len(nodes))
    return features, np.asarray(nodes)