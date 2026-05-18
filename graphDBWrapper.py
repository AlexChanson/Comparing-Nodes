from abc import ABC, abstractmethod
from typing import ClassVar
from config import *
from neo4j import Driver, GraphDatabase, basic_auth
from neo4j.exceptions import ServiceUnavailable, AuthError
from typing import Any, Collection, Dict, List, Optional, Set, Tuple


class I_GraphDBWrapper(ABC):
    @property
    @abstractmethod
    def connection_params(self) -> I_Config:
        """Abstract property requiring implementation."""
        pass

    @property
    @abstractmethod
    def driver(self) -> Driver:
        """Abstract property requiring implementation."""
        pass

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def execute_query(self):
        pass


    

class Neo4JWrapper(I_GraphDBWrapper):
    connection_params: ClassVar[Neo4JConnectionParams]
    driver: ClassVar[Driver]

    def __init__(self, neo4JConnectionParams):
        self.connection_params = neo4JConnectionParams
        # print(self.connection_params.toString())

        try:
            # Set up connection
            self.driver = GraphDatabase.driver(f"bolt://{self.connection_params.get("host")}", auth=basic_auth(self.connection_params.get("username"), self.connection_params.get("password")), encrypted=False)
            # Test connection
            self.driver.verify_connectivity()
            print("Connection to Neo4J successful!")

        except ServiceUnavailable:
            print("Neo4j service unavailable.")
        except AuthError:
            print("Authentication failed.")
        except Exception as e:
            print(f"Error: {e}")


    # Getters
    #####################################################
    def connection_params(self) -> Neo4JConnectionParams:
        return self.params

    def driver(self) -> Driver:
        return self.driver
    #####################################################
    

    def execute_query(self, query: str, parameters: dict = None) -> list:
        if parameters is None:
            parameters = {}
            
        try:
            with self.driver.session(database=self.connection_params.get("name")) as session:
                result = session.run(query, parameters)
                return [record.data() for record in result]
        except Exception as e:
            print(f"Query execution failed: {e}")
            exit(-1)