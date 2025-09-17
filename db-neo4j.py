from statistics import variance, pvariance

import numpy as np
from neo4j import GraphDatabase, basic_auth
from typing import Any, Dict, List, Optional
from scipy.stats import variation

import argparse
import pandas as pd

import utility
from many2many import aggregate_m2m_properties_for_label
from utility import outer_join_features
from validation import process_dataframe, export
from inDegrees import in_degree_by_relationship_type

import time

from typing import List, Dict, Optional

NUMERIC_TYPES = [
        # APOC meta cypher type names (Neo4j 4/5)
        "INTEGER", "FLOAT", "Number", "Long", "Double"
        # (If you want numeric lists too, add: "LIST OF INTEGER", "LIST OF FLOAT", etc.)
    ]

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

    def getDriver(self) -> GraphDatabase.driver:
        return self._driver

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

    def getAllPropFor(self,label)->List[Dict[str, Any]]:
        result = self.getNumericalProperties(label)


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



    def backtick_escape(self, label: str) -> str:
        """Escape backticks in a label so it can be safely backticked in Cypher."""
        return label.replace("`", "``")

    def build_cypher_bad(self, label: str, max_depth: Optional[int]) -> str:
        lbl = self.backtick_escape(label)
        depth = "" if max_depth is None else str(int(max_depth))

        return f"""
    // Numeric props from nodes + rels along functional (unique-outgoing) paths,
    // with path-based prefixes. 
    MATCH (n:{lbl})
    WITH n, head(labels(n)) AS rootLabel

    OPTIONAL MATCH p = (n)-[*0..]->(m)
    WHERE
    // only traverse unique-outgoing (functional) hops
    all(rel IN relationships(p)
    WHERE apoc.node.degree.out(startNode(rel), apoc.rel.type(rel)) = 1)
    // avoid revisiting nodes/edges
    AND size(relationships(p)) = size(apoc.coll.toSet(relationships(p)))
    AND size(nodes(p))        = size(apoc.coll.toSet(nodes(p)))

    WITH n, rootLabel, p, nodes(p) AS ns, relationships(p) AS rs, last(nodes(p)) AS m

    // Build prefix for END NODE of this path: City__REL1__Label2__REL2__Label3
    WITH n, rootLabel, ns, rs, m,
     reduce(pref = rootLabel,
            i IN (CASE WHEN size(rs)=0 THEN [] ELSE range(0, size(rs)-1) END) |
              pref + '__' + apoc.rel.type(rs[i]) + '__' + head(labels(ns[i+1]))
     ) AS nodePrefix

    // Map of numeric properties for the end node (prefixed by the full path)
    WITH n, rootLabel, ns, rs,
     apoc.map.fromPairs(
       [k IN keys(m)
        WHERE apoc.meta.cypher.type(m[k]) IN ['INTEGER','FLOAT','Number','Long','Double']
        | [nodePrefix + '_' + k, m[k]]
       ]
     ) AS nodeMap,
     ns AS ns_keep, rs AS rs_keep

// Build maps for EACH relationship's numeric properties along this path
WITH n, rootLabel, ns_keep AS ns, rs_keep AS rs, [nodeMap] AS collected
UNWIND (CASE WHEN size(rs)=0 THEN [] ELSE range(0, size(rs)-1) END) AS j
WITH n, rootLabel, ns, rs, collected,
     apoc.map.fromPairs(
       [k IN keys(rs[j])
        WHERE apoc.meta.cypher.type(rs[j][k]) IN ['INTEGER','FLOAT','Number','Long','Double']
        |
          // Prefix up to and including the j-th relationship:
          // City__REL1__Label2__...__RELj_prop
          [ reduce(pref = rootLabel,
                   i IN (CASE WHEN j=0 THEN [] ELSE range(0, j-1) END) |
                     pref + '__' + apoc.rel.type(rs[i]) + '__' + head(labels(ns[i+1]))
            ) + '__' + apoc.rel.type(rs[j]) + '_' + k,
            rs[j][k]
          ]
       ]
     ) AS relMap

// ---- separate aggregation from combination to avoid implicit-grouping error
WITH n, collected, collect(relMap) AS relMaps
WITH n, collected + relMaps AS mapsPerPath

// Now combine all paths for the same root node n
WITH n, collect(mapsPerPath) AS mapsPerPathList
WITH n, apoc.coll.flatten(mapsPerPathList) AS allMaps
RETURN
  id(n) AS rootId,
  apoc.map.mergeList(allMaps) AS mergedNumericProperties;
    """



    def detect_relationship_cardinalities(self):
        """
        Detects instance-based relationship cardinalities (one-to-one, one-to-many,
        many-to-one, many-to-many) per (relType, startLabelSet, endLabelSet).

        Args:
            driver: An open neo4j.Driver instance (Neo4j Python driver v4/v5).
            database: Optional database name (Neo4j Enterprise / multi-db).
            use_primary_label: If True, classify by head(labels(n)) instead of the full label set.

        Returns:
            List of dict rows with keys:
            ['relType','startLabels','endLabels','startPopulation','endPopulation',
             'minOut','maxOut','minIn','maxIn','cardinality']
        """
        query = """
        // Detect instance-based cardinalities: one-to-one, one-to-many, many-to-one, many-to-many
        MATCH (s)-[r]->(t)
        WITH type(r) AS relType,
             apoc.text.join(labels(s),'|')  AS startLabels,
             apoc.text.join(labels(t),'|')  AS endLabels
        WITH DISTINCT relType, startLabels, endLabels

        // For ALL start nodes with this label set, count outgoing edges of relType to endLabels
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

        // For ALL end nodes with this label set, count incoming edges of relType from startLabels
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

        result = self.execute_query(query)
        #return result
        return [rec['relType'] for rec in result if rec['cardinality']=='many-to-one'],[rec['relType'] for rec in result if rec['cardinality']=='many-to-many']

    def build_cypher_nodes(self, label: str, max_depth: Optional[int], many2one) -> str:
        """
        warning initially not on schema, was : all(rel...where apoc.node.degree.out(startNode(rel), apoc.rel.type(rel)) = 1)
        Build the Cypher query by inlining the label (labels can't be parameterized).
        Traverses only along steps where the current node has exactly one outgoing
        relationship of that type (functional chain).
        Keeps ONLY numeric properties via apoc.meta.cypher.type.
        """
        lbl = self.backtick_escape(label)
        depth = "" if max_depth is None else str(int(max_depth))

        # Note:
        # - Direction is OUTGOING (child -> parent). Flip to <-*0..- if your model is opposite.
        # - Properties are prefixed with the node's (first) label to avoid collisions.
        return f"""
        MATCH (n:`{lbl}`)
        OPTIONAL MATCH p = (n)-[*0..{depth}]->(m)
        WHERE
          all(rel IN relationships(p) WHERE
              type(rel) in {many2one})
        WITH n, collect(DISTINCT m) + n AS nodes
        WITH n,
             [x IN nodes |
                apoc.map.fromPairs(
                  [k IN keys(x)
                     WHERE apoc.meta.cypher.type(x[k]) IN {NUMERIC_TYPES}
                     | [head(labels(x)) + "_" + k, x[k]]
                  ]
                )
             ] AS maps
        RETURN id(n) AS rootId,
               apoc.map.mergeList(maps) AS mergedNumericProperties
        """
        # If you want numeric lists as well, replace the WHERE line by:
        # WHERE apoc.meta.cypher.type(x[k]) IN ["INTEGER","FLOAT","Number","Long","Double",
        #                                       "LIST OF INTEGER","LIST OF FLOAT","LIST OF Number",
        #                                       "LIST OF Long","LIST OF Double"]

    def build_cypher_edges(self, label: str, max_depth: Optional[int],many2one) -> str:
        """
        warning, initially  not on schema, was : all(rel ... where apoc.node.degree.out(startNode(rel), apoc.rel.type(rel)) = 1)
        Build the Cypher query by inlining the label (labels can't be parameterized).
        Traverses only along steps where the current node has exactly one outgoing
        relationship of that type (functional chain).
        Keeps ONLY numeric properties via apoc.meta.cypher.type.
        """
        lbl = self.backtick_escape(label)
        depth = "" if max_depth is None else str(int(max_depth))

        # Note:
        # - Direction is OUTGOING (child -> parent). Flip to <-*0..- if your model is opposite.
        # - Properties are prefixed with the node's (first) label to avoid collisions.
        return f"""
        MATCH (n:`{lbl}`)
        OPTIONAL MATCH p = (n)-[*0..{depth}]->(m)
        WHERE
          all(rel IN relationships(p) WHERE
               type(rel) in {many2one})
        WITH n,
            // Gather all relationships from all such paths, then deduplicate
        apoc.coll.toSet(apoc.coll.flatten(collect(relationships(p)))) AS rels
        WITH n,
             // Build one small map per relationship, keeping only numeric props.
            // Keys are prefixed with the relationship type to reduce collisions.
        [r IN rels |
        apoc.map.fromPairs(
          [k IN keys(r)
             WHERE apoc.meta.cypher.type(r[k]) IN ['INTEGER','FLOAT','Long','Double','Number']
             | [type(r) + "_" + k, r[k]]
          ]
        )
        ] AS maps
        WITH n, apoc.map.mergeList(maps) AS mergedNumericProperties
        WHERE size(keys(mergedNumericProperties)) > 0   // filter out empties
        RETURN id(n) AS rootId,
       mergedNumericProperties;
        """
        # If you want numeric lists as well, replace the WHERE line by:
        # WHERE apoc.meta.cypher.type(x[k]) IN ["INTEGER","FLOAT","Number","Long","Double",
        #                                       "LIST OF INTEGER","LIST OF FLOAT","LIST OF Number",
        #                                       "LIST OF Long","LIST OF Double"]

    def fetch_as_dataframe(self, out, label: str, max_depth: Optional[int],many2one) -> pd.DataFrame:
        #cypher = self.build_cypher(label, max_depth)
        #driver = GraphDatabase.driver(uri, auth=(user, password))
        #try:
        #    with driver.session() as session:
        #result = session.run(cypher)
        query=self.build_cypher_nodes(label, max_depth,many2one)
        result=self.execute_query(query)
        query_edge = self.build_cypher_edges(label, max_depth,many2one)
        result_edges = self.execute_query(query_edge)
        result=result+result_edges
        rows = []
        for rec in result:
            root_id = rec["rootId"]
            props = rec["mergedNumericProperties"] or {}
            row = {"rootId": root_id}
            row.update(props)
            rows.append(row)

        df = pd.DataFrame(rows).sort_values("rootId").reset_index(drop=True)
        df.to_csv(out, index=False)
        print(f"Saved {len(df)} rows to: {out}")
        print("Columns:", ", ".join(df.columns))
        return df

    def getAvgPropByElem(self):
        query=("CALL    {  MATCH(n)    WITH     n, [k IN keys(n) WHERE apoc.meta.cypher.type(n[k]) IN['INTEGER', 'FLOAT']] AS "
             +  " numericProps)   RETURN    avg(size(numericProps))    AS    avgNodeNumericProps    }"
             +  "    CALL    {        MATCH() - [r] - ()    WITH    r, [k IN keys(r) WHERE apoc.meta.cypher.type(r[k]) IN['INTEGER', 'FLOAT']]"
             +  " AS    numericProps    RETURN    avg(size(numericProps))    AS    avgRelNumericProps    }"
             +  "    RETURN    avgNodeNumericProps, avgRelNumericProps;")
        result = self.execute_query(query)
        return result


# Example usage:
if __name__ == "__main__":
    # adjust URI/user/password as needed
    #with Neo4jConnector("bolt://localhost:7687", "neo4j", "recommendations") as db:
    #with Neo4jConnector("bolt://localhost:7687", "neo4j", "uscongress") as db:
    with Neo4jConnector("bolt://localhost:7687", "neo4j", "airports") as db:
    #with Neo4jConnector("bolt://localhost:7687", "neo4j", "icijleaks") as db:
        #label="Actor"
        #label="Movie"
        #label="Director"
        #label="Legislator"
        #label="Officer"
        #label="Intermediary"
        #label="Entity"
        label="Airport"


        #True = remove lines with at least one null value
        NONULLS=True

        start_time = time.time()

        # finds relationship cardinalities
        manyToOne,manyToMany = db.detect_relationship_cardinalities()

        # get context and candidate indicators
        # get * relationships for label
        dfm2m = aggregate_m2m_properties_for_label(db.getDriver(), label, agg="sum", include_relationship_properties=True)

        # get 1 relationships for label (and save those to csv)
        out="sample_data/"+label+"_indicators.csv"
        df121=db.fetch_as_dataframe(out,label,10,manyToOne)

        #out join * and 1
        dftemp = outer_join_features(df121, dfm2m, id_left="rootId", id_right="node_id", out_id="out1_id")

        # get in degrees of label
        dfdeg = in_degree_by_relationship_type(db.getDriver(),label)

        # then outer join with dftemp
        dffinal = outer_join_features(dftemp, dfdeg, id_left="out1_id", id_right="nodeId", out_id="out_id")

        #validates and transform (scale) candidate indicators
        null_threshold=0.5
        distinct_low=0.000001
        distinct_high=0.96

        keep,report=process_dataframe(dffinal,null_threshold,distinct_low,distinct_high)

        processedIndicators="sample_data/"+label+"_indicators_processed.csv"
        processingReport="reports/"+label+"_indicators_processed.csv"

        # if we remove lines with at least one null
        if NONULLS:
            keep=utility.remove_rows_with_nulls(keep)
            processedIndicators = "sample_data/" + label + "_indicators_processed_nonulls.csv"

        export(keep,report,processedIndicators,processingReport)


        end_time = time.time()
        timings = end_time - start_time
        print('Completed in ', timings, 'seconds')

        #print(db.getNumericalProperties('Airport'))
        #print(db.getRelationCardinality())
        #print(db.getValidProperties('Airport'))

        #f,m = db.createDatasetForLabel('Airport')
        #print(f)
        #print(m)

        #db.getDegreeOfRelationForLabel('Airport')

        #print(db.getBestPageRank('Airport', 'ROUTE_TO'))
