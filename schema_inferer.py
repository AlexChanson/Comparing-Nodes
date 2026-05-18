from abc import ABC, abstractmethod
from typing import ClassVar
from graphDBWrapper import *

class I_Schema_Inferer(ABC):
    
    @abstractmethod
    def __init__(self, graphDbWrapper: I_GraphDBWrapper):
        pass
    
    @property
    @abstractmethod 
    def graphDbWrapper(self) -> I_GraphDBWrapper:
        pass

    @abstractmethod 
    def detect_relationships_cards(self) -> dict:
        pass



class Neo4J_Side_Schema_Inferer(I_Schema_Inferer):
    graphDbWrapper: ClassVar[Neo4JWrapper]

    def __init__(self, graphDbWrapper: Neo4JWrapper):
        self.graphDbWrapper = graphDbWrapper


    def detect_relationships_cards(self):
        """Returns two arrays, one containing all the One-to-One relationships,
         and another containing the One-to-Many"""
        

        parametres = {
        "allowed_labels": ["Team"]
        }
        query = """
        MATCH (s)-[r]->(t)
        WITH type(r) AS relType,
             apoc.text.join(labels(s),'|')  AS startLabels,
             apoc.text.join(labels(t),'|')  AS endLabels
        WITH DISTINCT relType, startLabels, endLabels

        CALL {
          WITH relType, startLabels, endLabels
          MATCH (sx)
          WHERE apoc.text.join(labels(sx),'|')  = startLabels
          OPTIONAL MATCH (sx)-[rx]->(tx)
          WHERE type(rx) = relType
            AND apoc.text.join(labels(tx),'|')  = endLabels
          WITH sx, count(rx) AS outCnt
          RETURN
            min(outCnt) AS minOut,
            max(outCnt) AS maxOut,
            count(sx)   AS startPopulation
        }

        CALL {
          WITH relType, startLabels, endLabels
          MATCH (ty)
          WHERE apoc.text.join(labels(ty),'|')  = endLabels
          OPTIONAL MATCH (sy)-[ry]->(ty)
          WHERE type(ry) = relType
            AND apoc.text.join(labels(sy),'|')  = startLabels
          WITH ty, count(ry) AS inCnt
          RETURN
            min(inCnt) AS minIn,
            max(inCnt) AS maxIn,
            count(ty)  AS endPopulation
        }

        RETURN
          relType,
          startLabels,
          endLabels,
          startPopulation,
          endPopulation,
          minOut, maxOut, minIn, maxIn,
          CASE
            WHEN minOut >= 1 AND maxOut <= 1 AND minIn >= 1 AND maxIn <= 1 THEN 'one-to-one'
            WHEN minOut >= 0 AND maxIn <= 1 THEN 'one-to-many'
            WHEN maxOut <= 1 AND minIn >= 0 THEN 'many-to-one'
            ELSE 'many-to-many'
          END AS cardinality
        ORDER BY relType, startLabels, endLabels
        """

        result = self.graphDbWrapper.execute_query(query)
        return {"one-to-one" : [rec['relType'] for rec in result if rec['cardinality'] == 'many-to-one'], 
                "one-to-many": [rec['relType'] for rec in result if rec['cardinality'] == 'many-to-many'
        ]}
    
    # Getters
    #####################################################
    def graphDbWrapper(self):
        return self.graphDbWrapper
    #####################################################

class Python_Side_Schema_Inferer(I_Schema_Inferer):
    # Should call the PG Schema inferer algorithm
    def __init__(self):
        super().__init__()
        
