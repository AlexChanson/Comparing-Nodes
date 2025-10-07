# Extracting node comparison insights for the interactiveexploration of property graphs

### Requirements
This code has been tested with Python 3.12, and Neo4J 5.x, to install the required packages we recommend the use of a pythin virtual environement.
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
#### Using custom datasets
Sample data is provided in this repository. To use your own data you'll need to setup the graph in a local Neo4J database (using Neo4J desktop for exemple) in order to process the indicator extraction. We only tested with Neo4J 5.x on Mac OS and Linux. 
## Indicators extraction

To extract indicators from property graphs use the `db-neo4j.py` script.
Several variables need to be set in the script :
 - dict_databases_homes : the path to the Neo4J DBs installation(s)
 - dict_databases_labels : the labels of the graph to process for each database
 - null_threshold, distinct_low, distinct_high, correlation_threshold : thresholds to discard indicators 
 - NONULLS : boolean whether or not to keep nodes with null values

## Solving the Partiton/Clustering problem

To run the clustering and indicator partition problem use the main.py script. This script uses command line arguments to specify datasets and parameters:
 - `--k`: the number of clusters desired
 - `--steps`: To limit the local search steps
 - `--method`: ls : local search, exp : full tree enumeration, sls: 'smart' start local search
 - `--dataset`: Use one of : iris (debug only), airports, movies, directors, actors or custom
 - `--path`: the path to the dataset (custom only, must be a csv file with a header)
 - `--delimiter` :  for custom dataset


