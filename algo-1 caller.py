import argparse
import json
import traceback
from utility import json_config_file_parser
from config import *
from exceptions import *
from graphDBWrapper import *
from algorithm_one import *


if __name__ == "__main__":

    # Getting args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', default=None, type=str)
    args = parser.parse_args()
    
    # Getting config parameters
    parameters = json_config_file_parser(args.config)
    try:
        if parameters is None:
            raise ConfigNotValidError("Can't find the configuration file !")
        print("Process called with parameters : ", parameters)
    except Exception as e:
        print(e)
        exit(-1)

    # Creating the matching database config class
    database_type = None
    database_config = None
    wrapper = None
    try :
        match parameters["database"]["type"]:
            # Neo4J Property Graph Database
            case "Neo4J":
                database_type = "Neo4J"
                # host, username, pwd, name
                database_config = Neo4JConnectionParams(parameters["database"]["host"],
                                                        parameters["database"]["username"],
                                                        parameters["database"]["password"], 
                                                        parameters["database"]["name"])
                # Creating the Neo4jWrapper (Connection)
                wrapper = Neo4JWrapper(database_config)
            case _:
                raise ConfigNotValidError("Error while trying get database's configuration from configuration file !\n")
    except Exception as e:
        print(e)
        exit(-1)


    # Creating the config for Algorithm-1
    algo1_type = None
    algo1_config = None
    algo1 = None
    try:
        match parameters["algorithm-1"]["type"]:
            case "base":
                algo1_type = "base"
                # target_labels, discarded, min_var, max_var, min_density, correlation_threshold
                algo1_config = Basic_Algo1Params(parameters["algorithm-1"]["target_labels"],
                                                parameters["algorithm-1"]["discarded"], 
                                                parameters["algorithm-1"]["min_var"], 
                                                parameters["algorithm-1"]["max_var"],
                                                parameters["algorithm-1"]["min_density"],
                                                parameters["algorithm-1"]["correlation_threshold"])
                # Set up of the Algo1
                algo1 = Basic_Algo1(algo1_config, wrapper)
                
            case _:
                raise ConfigNotValidError("Error while trying get Algorithm-1's configuration from configuration file !\n")
    except Exception as e:
        print(traceback.format_exc())
        exit(-1)


    algo1.collect_indicators()

    print("ok")



    
