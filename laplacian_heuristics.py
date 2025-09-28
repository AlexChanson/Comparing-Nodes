from sklearn.preprocessing import StandardScaler
from scipy.stats import kurtosis, skew
from skfeature.function.similarity_based import lap_score
from skfeature.utility import construct_W
from sklearn.cluster import KMeans

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kneed import KneeLocator

def build_feature_groups(df, column, nclusters):
    X = df[[column]].values  # 2D array shape (n_samples, 1)

    kmeans = KMeans(n_clusters=nclusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(X)

# method to define an elbow
def find_elbow_threshold(df, column, visualize=False):
    # Step 1: Sort values
    sorted_vals = np.sort(df[column].values)

    # Step 2: Normalize to [0,1] for stability
    #x = np.linspace(0, 1, len(sorted_vals))
    #y = (sorted_vals - sorted_vals.min()) / (sorted_vals.max() - sorted_vals.min())
    x = np.arange(0, len(sorted_vals))
    y = sorted_vals

    # Step 3: Find elbow using curvature
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label='Curve')
    knee = KneeLocator(x, y, curve='convex', direction='increasing')
    #print(len(sorted_vals))
    if knee.elbow==len(sorted_vals)-1:
        knee = KneeLocator(x, y, curve='concave', direction='increasing')
    elbow_index = knee.knee
#    threshold = sorted_vals[int(elbow_index * len(sorted_vals))] if elbow_index else None

    print(knee.all_knees_y)

    # Step 4: Optional visualization
    if visualize:
        plt.plot(x, y, label='Normalized values')
        if elbow_index:
            plt.axvline(elbow_index, color='red', linestyle='--', label='Elbow')
        plt.title(f'Elbow detection for {column}')
        plt.legend()
        plt.show()

    return knee.knee_y
#    return threshold


def score(file,ngroups=2):
    # Load CSV into a DataFrame
    #data = pd.read_csv('./data/Movie_indicators_processed_nonulls.csv') # Airport_indicators_processed_nonulls
    data = pd.read_csv(file) # Airport_indicators_processed_nonulls
    X = data.drop(data.columns[0], axis=1) # drop id columns

    # Preview the data
    print(data.head())

    # Load and standardize Iris dataset
    #data = load_iris()
    #X = pd.DataFrame(data.data, columns=data.feature_names)
    X_std = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

    # Initialize metrics
    metrics = {
        #'variance_ratio': [],
        'cv': [],
        'cv_pairwise': [],
        'bimodality': [],
        'laplacian_score': []
    }

    # Total variance
    #total_var = X_std.var().sum()

    # Compute metrics per feature
    for col in X_std.columns:
        x = X_std[col].values

        # OLD 1. Variance ratio
        #var_ratio = np.var(x) # / total_var
        #metrics['variance_ratio'].append(var_ratio)

        # 1. coefficient of variation
        cv = np.std(x) / np.mean(x)
        metrics['cv'].append(cv)

        # 2. Coefficient of variation over pairwise distances
        diffs = np.diff(np.sort(x))
        cv_pairwise = np.std(diffs) / np.mean(diffs)
        metrics['cv_pairwise'].append(cv_pairwise)

        # 3. Bimodality coefficient
        gamma = skew(x)
        kappa = kurtosis(x, fisher=False)
        bimodality = (gamma**2 + 1) / kappa
        metrics['bimodality'].append(bimodality)

    # 4. Spectral weights (Laplacian Score)
    W = construct_W.construct_W(X_std.values, neighbor_mode='knn', k=5)
    lap_scores = lap_score.lap_score(X_std.values, W=W)
    metrics['laplacian_score'] = list(lap_scores)

    # Create DataFrame
    metrics_df = pd.DataFrame(metrics, index=X_std.columns)

    # define the metric / column to use for the heuristic
    col_metric = 'laplacian_score'

    # Sort by Laplacian Score and keep feature names as a column
    sorted_df = metrics_df.sort_values(by=col_metric, ascending=True)
    sorted_df = sorted_df.reset_index().rename(columns={'index': 'feature'})

    # Compute correlation matrix
    correlation_matrix = metrics_df.corr()

    # Display results
    print("\n Feature Metrics:")
    print(sorted_df.round(3))
    #print(metrics_df.round(3))

    print("\n Correlation Between Metrics:")
    print(correlation_matrix.round(3))

    #build_feature_groups(sorted_df, col_metric,ngroups)

    #pd.set_option('display.max_columns', None)
    #print(sorted_df)
    # define a cutting point in the selected metric / column
    threshold = find_elbow_threshold(sorted_df, col_metric, visualize=True)
    #
    # Split the column based on threshold
    sorted_df['split'] = sorted_df[col_metric] > threshold
    print(sorted_df)
    dictResult={}
    dictResult['clustering'] = list(sorted_df.loc[sorted_df['split'] ==False]['feature'])
    #print(dictResult)

    forComparison=sorted_df[sorted_df['split']==True]
    if forComparison.empty:
        dictResult['comparison']=[]
        dictResult['unused'] = []
    else:
        #forComparison=forComparison.drop(columns=['cluster'])
        col_metric = 'cv'
        #build_feature_groups(forComparison, col_metric, ngroups)
        threshold = find_elbow_threshold(forComparison, col_metric, visualize=True)
        forComparison['split2'] = forComparison[col_metric] > threshold
        print(forComparison)
        dictResult['comparison'] = list(forComparison.loc[forComparison['split2'] == True]['feature'])
        dictResult['unused'] = list(forComparison.loc[forComparison['split2'] == False]['feature'])

    print(dictResult)
    return dictResult


if __name__ == "__main__":
    #score('sample_data/Actor_indicators_processed_nonulls.csv',2)
    #score('sample_data/Director_indicators_processed_nonulls.csv',2)
    #score('sample_data/Intermediary_indicators_processed_nonulls.csv',2)
    #score('sample_data/Country_indicators_processed_nonulls.csv',2)
    #score('sample_data/Airport_indicators_processed_nonulls.csv',2)
    #score('sample_data/Movie_indicators_processed_nonulls.csv',2)
    score('sample_data/Entity_indicators_processed_nonulls.csv',2)
