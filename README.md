# Extracting node comparison insights for the interactive exploration of property graphs

### Requirements
This code has been tested with Python 3.12, and Neo4J 5.x, to install the required packages we recommend the use of a python virtual environement.
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
#### Using custom datasets
Sample data are provided in this repository. To use your own data, you need to setup the graph in a local Neo4J database (using Neo4J desktop for example) in order to process indicator extraction. We only tested with Neo4J 5.x on Mac OS and Linux. 

## Indicators extraction

To extract indicators from property graphs, use the `collect_indicators.py` script. This script uses command line arguments to specify the Neo4J database and parameters:
 - `-r`: the number of runs
 - `-dh`: the threshold for acceptable variance (high)
 - `-dl`: the threshold for acceptable variance (low)
 - `-c`: the threshold for non-redundancy (Pearson's correlation)
 - `-n`: the threshold for acceptable density
 - `-u`: a list of suffixes for discarding properties
 - `--pushdown`: if unwanted properties, acceptable density (validation) are pushed down indicator collection
 - `--keep-nulls`: to remove nodes with at least one null value
 - `--create-index`: to create all indices on numerical properties
 - `config`: the json file for the Neo4j database configuration (see examples in the config subdirectory)


## Solving the Partiton/Clustering problem

To run the clustering and indicator partition heuristics, use the `main.py`  script. This script uses command line arguments to specify datasets and parameters:
 - `--k`: the number of clusters desired
 - `--steps`: To limit the local search steps
 - `--method`: ls : local search, exp : full tree enumeration, sls: 'simple' start local search
 - `--dataset`: Use one of : iris (debug only), airports, movies, directors, actors or custom
 - `--path`: the path to the dataset (custom only, must be a csv file with a header)
 - `--delimiter` :  for custom dataset


