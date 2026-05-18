from abc import ABC, abstractmethod
from typing import ClassVar


class I_Config(ABC):
    @property
    @abstractmethod
    def params(self) -> dict:
        """Abstract property requiring implementation."""
        pass

    @abstractmethod
    def __init__(self):
        pass

    def get(self, key):
        return self.params[key]

    def set(self, key, value):
        self.params[key] = value

    def toString(self) -> str:
        str = ""
        for key, value in self.params.items():
            str += f"{key} : {value} \n"
        return str

#
class Neo4JConnectionParams(I_Config):
    params: ClassVar[dict] = {"host": None, "username": None, "password": None, "name": None}


    def __init__(self, host, username, pwd, name):
        self.params = {"host": host, "username": username, "password": pwd, "name": name}

    # Getters
    #####################################################
    def params(self):
        return self.params
    #####################################################

class I_Algo1Params(I_Config):


    @abstractmethod
    def __init__(self):
        pass


class Basic_Algo1Params(I_Algo1Params):

    params: ClassVar[dict] = {
        "target_labels": None,
        "discarded": None,
        "min_var": None,
        "max_var": None,
        "min_density": None,
        "correlation_threshold": None,
    }

    def __init__(self, target_labels, discarded, min_var, max_var, min_density, correlation_threshold):
        self.params = {
            "target_labels" : target_labels,
            "discarded": discarded,
            "min_var": min_var,
            "max_var": max_var,
            "min_density": min_density,
            "correlation_threshold": correlation_threshold,
        }
