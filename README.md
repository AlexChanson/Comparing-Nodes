# Extracting node comparison insights for the interactive exploration of property graphs

## Requirements and setup guide

This code has been tested with Python 3.12 and Neo4J 5.x.

A Java Development Kit (JDK) version 21 or 25 is also required to run the local Neo4J database instance. At the time of writing, more recent versions (such as JDK 26) are not supported and may prevent Neo4J Desktop from starting correctly.

### 1. Install a code editor

Install VS Code (recommended), or any editor of your choice:

https://code.visualstudio.com/

### 2. Install Python 3.12

Install the latest security release of Python 3.12 from the official Python website:

https://www.python.org/downloads/

Click on the **Download** button, select your operating system, and install a Python 3.12 version.

### 3. Install the Java Development Kit (JDK)

Install JDK 25 from the official Oracle website:

https://www.oracle.com/fr/java/technologies/downloads/#java25

### 4. Install Neo4J Desktop

Download and install Neo4J Desktop:

https://neo4j.com/download/

### 5. Install VS Code extensions

In VS Code, install the following extensions:

- Python (official Microsoft extension):
  https://marketplace.visualstudio.com/items?itemName=ms-python.python

- Pylance:
  https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance

Pylance provides intelligent code analysis and autocompletion for Python projects.

### 6. Create a Neo4J database instance

In Neo4J Desktop:

- Create a new instance
- Give it a name
- Select a Neo4J version if not already selected automatically
- Optionally import a `.dump` file containing graph data (for example the airports dataset)

### 7. Install the APOC plugin

After creating the instance:

- Click on the button represented by the three dots
- Open the **Plugins** section
- Install the plugin named **APOC**

This plugin is required to execute some graph algorithms.

You can explore the graph by clicking on the **Explore** section in the Neo4J Desktop sidebar.

### 8. Clone the repository

Create a directory where you want to store the project.

In VS Code:

- Open the directory
- Open a terminal
- Clone the repository with the following command:

```bash
git clone https://github.com/AlexChanson/Comparing-Nodes.git
```

### 9. Create the `reports` directory

After cloning the repository, create a directory named `reports` at the root of the project:

```bash
mkdir reports
```

This directory is required by the `collect_indicators.py` script to store generated CSV reports, validation summaries, execution metrics, and LaTeX exports used for performance analysis and scientific reporting.

If the directory does not exist, the script may fail when exporting results.

### 10. Create the virtual environment and install dependencies

Before creating the virtual environment, make sure that VS Code is using the correct Python interpreter (Python 3.12). Otherwise, the virtual environment may use another Python version installed on your machine (for example the most recent one), which may not be compatible with this project.

To select the correct Python interpreter in VS Code:

- Open the Command Palette:
  - macOS: `Cmd + Shift + P`
  - Windows/Linux: `Ctrl + Shift + P`
- Type:

```text
Python: Select Interpreter
```

- Select a Python 3.12 interpreter from the list

Then open the project in VS Code and execute the following commands in the integrated terminal:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

To verify that the virtual environment is correctly using Python 3.12, you can execute one of the following commands:

```bash
python3 --version
```

or

```bash
python --version
```

For example, this may return:

```text
Python 3.12.9
```

### 11. Configure the Neo4J connection

In the JSON configuration file located in the `configs` directory:

- Replace the `home` field with the path to your Neo4J database instance
- Fill the `username` and `password` fields using the credentials chosen during the Neo4J instance setup

### 12. Manage the Neo4J database instance

Before executing the `collect_indicators.py` script, make sure that the Neo4J database instance is not currently running.

To control the database instance:

- Retrieve the path to the Neo4J instance
- Append `/bin/neo4j` to the path
- Use one of the following commands:
  - `start` to start the instance
  - `stop` to stop the instance
  - `status` to check whether the instance is running

The expected state before running the script is:

```text
Neo4j is not running
```

### 13. Example command

Example command to check the database status:

```bash
"neo4j_instance_path/bin/neo4j" status
```

---

### Using custom datasets

Sample data are provided in this repository. To use your own data, you need to set up the graph in a local Neo4J database (using Neo4J desktop, for example) in order to process indicator extraction. We only tested with Neo4J 5.x on Mac OS and Linux.

## Indicators extraction

To extract indicators from property graphs, use the `collect_indicators.py` script. This script uses command-line arguments to specify the Neo4J database and parameters:

* `config`: the json file for the Neo4j database configuration (see examples in the config subdirectory)
* `-r` (or `--runs`): the number of runs
* `-a` (or `--agg-config`): path to a CSV file for aggregation configuration
* `-dl` (or `--distinct-low`): the threshold for acceptable variance (low)
* `-dh` (or `--distinct-high`): the threshold for acceptable variance (high)
* `-c` (or `--correlation-threshold`): the threshold for non-redundancy (Pearson's correlation)
* `-n` (or `--null-threshold`): the threshold for acceptable density
* `-u` (or `--unwanted-suffixes`): a list of suffixes for discarding properties
* `--pushdown`: if unwanted properties and acceptable density validation are pushed down during indicator collection
* `--keep-nulls`: remove lines with at least one null value
* `--create-index`: create all indices on numerical properties

### Example command

You can run the indicator extraction using the following command:

```bash
python3 collect_indicators.py configs/airports.json \
    --runs 1 \
    --distinct-low 0.000001 \
    --distinct-high 1 \
    --correlation-threshold 0.98 \
    --null-threshold 0.1 \
    --unwanted-suffixes _id _zipcode \
    --pushdown \
    --create-index
```

## Solving the Partition/Clustering problem

To run the clustering and indicator partition heuristics, use the `main.py` script. This script uses command-line arguments to specify datasets and parameters:

* `-k`: the number of clusters desired
* `-steps`: to limit the local search steps
* `-method`: `ls` for local search, `exp` for full tree enumeration, `sls` for 'simple' start local search
* `-dataset`: use one of: `iris` (debug only), `airports`, `movies`, `directors`, `actors`, or `custom`
* `-path`: the path to the dataset (custom only, must be a CSV file with a header)
* `-delimiter`: delimiter used for custom datasets

### Example command

You can run the clustering process on a custom dataset using the following command:

```bash
python3 main.py \
    --dataset custom \
    --path data/Player_indicators_processed_nonulls.csv \
    --method ls \
    --k 3 \
    configs/soccer.json
```