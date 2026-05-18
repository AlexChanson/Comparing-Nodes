from abc import ABC, abstractmethod
import traceback
from config import I_Algo1Params, Basic_Algo1Params
from typing import ClassVar
from schema_inferer import *
from graphDBWrapper import *



class I_Algo_1(ABC):
    
    @property
    @abstractmethod
    def params(self) -> I_Algo1Params:
        """Abstract property requiring implementation."""
        pass

    @property
    @abstractmethod
    def wrapper(self) -> I_GraphDBWrapper:
        """Abstract property requiring implementation."""
        pass

    @abstractmethod
    def __init__(self):
        pass

    
    @abstractmethod
    def collect_indicators(self):
        pass

class Basic_Algo1(I_Algo_1):
    params: ClassVar[Basic_Algo1Params]
    wrapper: ClassVar[I_GraphDBWrapper]

    def __init__(self, params, wrapper):
        self.params = params
        self.wrapper = wrapper

    def collect_indicators(self):
    
        
        # Compute schema inference
        try :
            # Checking for the wrapper used
            match type(self.wrapper).__name__:
                case Neo4JWrapper.__name__:
                    # Create Neo4J Inferer
                    inferer = Neo4J_Side_Schema_Inferer(self.wrapper)
                case _:
                    raise Exception("Wrapper not well defined !")
        except Exception as e:
            print(traceback.format_exc())
            exit(-1)
        #  
        cards = inferer.detect_relationships_cards()
        print("Cards : ", cards)

        one2one = cards["one-to-one"]
        one2many = cards["one-to-many"]

        # Getting all the one to one 

        #
        # 3 : Filter on One-to-One properties 
        #
        # 4 : Apply density filter on pre-agg One-to-Many vars
        #
        # 5 : Aggregate One-to-Many vars, and apply filters (Variance, correlation)
        #
        # 6 : Return indicators
        print("ok")
        return None
    
    # Getters
    #####################################################
    def params(self) -> Basic_Algo1Params:
        return self.params
    
    def wrapper(self) -> I_GraphDBWrapper:
        return self.wrapper
    #####################################################
    
