from statistics import variance, pvariance
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
        #print("on "+property)
        #print(len(tabVal))
        #print(len(set(tabVal)))
        # then compute variance
        #print(tabVal)
        return len(tabVal),len(set(tabVal)),variance(tabVal),pvariance(tabVal),variation(tabVal)

    def getValidProperties(self,label):
        result=[]
        tabProp=self.getNumericalProperties(label)
        for p in tabProp:
            nb,nbd,v,pv,cv=self.varianceOfProperty(p,label)
            #print("for property: "+p+ " we have cv:" +str(cv))
            #if cv > 0 and cv < 1:
            print(p,nb,nbd)
            if nbd<nb and nbd>1:
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
        return carinalities


# Example usage:
if __name__ == "__main__":
    # adjust URI/user/password as needed
    with Neo4jConnector("bolt://localhost:7687", "neo4j", "airports") as db:

        result=db.getNumericalProperties('Airport')
        print(result)

        print(db.getRelationCardinality())

        print(db.getValidProperties('Airport'))