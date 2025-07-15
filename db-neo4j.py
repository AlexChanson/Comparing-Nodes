from statistics import variance, pvariance

import numpy as np
from neo4j import GraphDatabase, basic_auth
from typing import Any, Dict, List, Optional
from scipy.stats import variation


class Neo4jConnector:
    """
    A simple Neo4j connection manager and query runner.
    """

    def __init__(
            self,
            uri: str,
            user: str,
            password: str,
            encrypted: bool = False,
            **driver_kwargs: Any
    ):
        """
        Initialize the driver.

        :param uri: Neo4j URI, e.g. "bolt://localhost:7687" or "neo4j+s://<host>"
        :param user: Username
        :param password: Password
        :param encrypted: Whether to use encryption (defaults to False)
        :param driver_kwargs: Any additional kwargs passed to GraphDatabase.driver()
        """
        self._driver = GraphDatabase.driver(
            uri,
            auth=basic_auth(user, password),
            encrypted=encrypted,
            **driver_kwargs
        )

    def close(self) -> None:
        """
        Close the underlying driver connection.
        """
        self._driver.close()

    def execute_query(
            self,
            query: str,
            parameters: Optional[Dict[str, Any]] = None,
            database: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Run a Cypher query and return a list of records as dictionaries.

        :param query: Cypher query string
        :param parameters: Query parameters dict
        :param database: Optional database name (for multi‐db setups)
        :return: List of record dictionaries
        """
        with self._driver.session(database=database) as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    def __enter__(self) -> "Neo4jConnector":
        """Enable use as a context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Ensure the driver is closed on exit."""
        self.close()

    def getNumericalProperties(self, label):
        """returns the names of numerical properties of nodes with label"""
        result=[]
        nodes = self.execute_query("MATCH (n:"+label+") RETURN apoc.meta.cypher.types(n)")
        for n in nodes:
            for p in n['apoc.meta.cypher.types(n)']:
                if n['apoc.meta.cypher.types(n)'][p]=='INTEGER' or n['apoc.meta.cypher.types(n)'][p]=='FLOAT':
                    result.append(p)
        return set(result) #ugly fixme

    def varianceOfProperty(self, property,label):
        # get variance indicators of property for nodes of label
        # returns nb values, nb distinct values, variance of sample, variance of population, coefficient of variation
        # first need all the values
        values = self.execute_query("MATCH (n:" + label + ") RETURN n."+property)
        str="n."+property
        tabVal=[]
        for v in values:
            if v[str]!=None:
                tabVal.append(v[str])
        return tabVal,len(tabVal),len(set(tabVal)),variance(tabVal),pvariance(tabVal),variation(tabVal)

    def getValidProperties(self,label):
        result=[]
        tabProp=self.getNumericalProperties(label)
        for p in tabProp:
            tab,nb,nbd,v,pv,cv=self.varianceOfProperty(p,label)
            if nbd<nb and nbd>1: #fixme check better ways of validating
                result.append(p)
        return result

    def getRelationCardinality(self):
        query = ("MATCH (x)-[r]->(y) " 
                "WITH type(r) AS name, head(labels(x)) AS label_x, head(labels(y)) " 
                "AS label_y, count(distinct x) AS count_x, count(distinct y) "
                "AS count_y, count(distinct r) AS count_r "
                "RETURN name,"
                "    CASE WHEN count_y < count_r THEN label_x ELSE label_y END AS source,"
                "    CASE WHEN count_y < count_r THEN label_y ELSE label_x END AS target,"
                "    count_x < count_r AND count_y < count_r AS is_many_to_many")
        carinalities=self.execute_query(query)
        dictRel={}
        dictRel['manyToMany']=[]
        dictRel['oneToMany']=[]
        for c in carinalities:
            if c['is_many_to_many']==True:
                dictRel['manyToMany'].append(c['name'])
            else:
                dictRel['oneToMany'].append(c['name'])
        return dictRel

    def createDatasetForLabel(self,label):
        tabProp=self.getValidProperties(label)
        str='elementId(n),'
        for p in tabProp:
            str=str+'n.'+p+','
        str=str[:-1]
        values = self.execute_query("MATCH (n:" + label + ") RETURN "+str)
        # transform into features + matrix
        # 1. Extract feature names from the keys of the first dict, stripping the "n." prefix
#        features = [key.split('.', 1)[1] for key in values[0].keys()]
        features = [key for key in values[0].keys()]

        # 2. Build the matrix of values, row by row
        matrix = [
            [row.get(f"{feat}") for feat in features]
            for row in values
        ]
        features2=[]
        for f in features:
            if f.startswith("n."):
                f2=f.replace("n.","")
            if f.startswith("elementId"):
                f2="elementId"
            features2.append(f2)
        return features2, np.asarray(matrix)

    def getDegreeOfRelationForLabel(self,label):
        #returns degree of type for nodeId
        #tabProp=self.getValidProperties(label)
        str=''
        types = self.execute_query("call db.relationshipTypes()")
        rel=[ r['relationshipType'] for r in types ]
        print(rel)
        dictRel={}
        for r in rel:
            degrees = self.execute_query("MATCH (n:"+label+")-[r:"+r+"]-() RETURN elementId(n),count(r) AS count")
            #print(degrees)
            for d in degrees:
                if d['elementId(n)'] in dictRel:
                    dictRel[d['elementId(n)']][r] = d['count']
                else:
                    dictRel[d['elementId(n)']]={}

        print(dictRel)
        rel.append("elementId")
        return rel, dictRel


    def getBestPageRank(self, label, relationship):
        # retrieve the id of the node with highest page rank value
        # first remove graph if exists
        self.execute_query("CALL gds.graph.drop('myGraph', false) YIELD graphName;")
        queryGraph="MATCH (source:"+label+")-[:"+relationship+"]->(target:"+label+") RETURN gds.graph.project(  'myGraph',  source,  target);"
        queryResult=("CALL gds.pageRank.stream('myGraph') YIELD nodeId, score RETURN elementId(gds.util.asNode(nodeId)) AS elementId, score "
                     "ORDER BY score DESC, elementId ASC limit 1;")
        self.execute_query(queryGraph)
        result = self.execute_query(queryResult)
        return result[0]['elementId']

# Example usage:
if __name__ == "__main__":
    # adjust URI/user/password as needed
    with Neo4jConnector("bolt://localhost:7687", "neo4j", "airports") as db:

        #print(db.getNumericalProperties('Airport'))
        print(db.getRelationCardinality())
        #print(db.getValidProperties('Airport'))

        f,m = db.createDatasetForLabel('Airport')
        #print(f)
        #print(m)

        #db.getDegreeOfRelationForLabel('Airport')

        print(db.getBestPageRank('Airport', 'ROUTE_TO'))