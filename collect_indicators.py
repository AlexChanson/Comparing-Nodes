from statistics import variance, pvariance

import numpy as np
from neo4j import GraphDatabase, basic_auth
from typing import Any, Dict, List, Optional

from scipy.stats import variation

import pandas as pd

import analyzeIndicatorDevisingTimes
import utility
from experiments import heuristics
from many2many import aggregate_m2m_properties_for_label
from utility import outer_join_features
from validation import process_dataframe, export, remove_correlated_columns, drop_columns_by_suffix_with_report
from inDegrees import in_degree_by_relationship_type
import time
from typing import List, Dict, Optional
from orchestrate_neo4j import start_dbms, stop_dbms, DbSpec, stop_current_dbms

from laplacian_heuristics import score
from contextualization import schema_hops_from_label,weight_df_by_schema_hops

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


    def backtick_escape(self, label: str) -> str:
        """Escape backticks in a label so it can be safely backticked in Cypher."""
        return label.replace("`", "``")

    def detect_relationship_cardinalities(self):
        """
        Detects instance-based relationship cardinalities (one-to-one, one-to-many,
        many-to-one, many-to-many) per (relType, startLabelSet, endLabelSet).

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

    def fetch_as_dataframe(self, out, label: str, max_depth: Optional[int],many2one,checkedges=True) -> pd.DataFrame:
        query=self.build_cypher_nodes(label, max_depth,many2one)
        result=self.execute_query(query)
        if checkedges:
            query_edge = self.build_cypher_edges(label, max_depth,many2one)
            result.extend(self.execute_query(query_edge))
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

    def getAvgPropByElem(self,label):
        query=("CALL    {  MATCH(n:"+label+")    WITH     n, [k IN keys(n) WHERE apoc.meta.cypher.type(n[k]) IN['INTEGER', 'FLOAT']] AS "
             +  " numericProps   RETURN    avg(size(numericProps))    AS    avgNodeNumericProps    }"
             +  "    CALL    {        MATCH() - [r] - ()    WITH    r, [k IN keys(r) WHERE apoc.meta.cypher.type(r[k]) IN['INTEGER', 'FLOAT']]"
             +  " AS    numericProps    RETURN    avg(size(numericProps))    AS    avgRelNumericProps    }"
             +  "    RETURN    avgNodeNumericProps, avgRelNumericProps;")
        result = self.execute_query(query)
        return result

    import pandas as pd







if __name__ == "__main__":
    current_time = time.localtime()
    formatted_time = time.strftime("%d-%m-%y:%H:%M:%S", current_time)
    fileResults = 'reports/results_' + formatted_time + '.csv'
    column_names = ['run','database', 'N','E', 'label', 'indicators#', 'nodes#', 'avgLabelProp', 'time_Cardinalities', 'time_Indicators','time_Validation','time_Partition','partition']
    dfresults = pd.DataFrame(columns=column_names)

    #  URI/user/password
    uri="bolt://localhost:7687"
    user="neo4j"
    dict_databases_labels={#"airports":["Airport","Country","City"],
                           "airportnew": ["Airport", "Country", "City"],
                           "recommendations":["Actor","Movie","Director"],
                            "icijleaks":["Entity", "Intermediary", "Officer"]
                            #,"icijleaks": ["Intermediary"]
                           }
    dict_databases_passwords={"airports":"airports", "airportnew":"airportnew", "recommendations":"recommendations", "icijleaks":"icijleaks"}
    dict_databases_homes={"airports":"/Users/marcel/Library/Application Support/Neo4j Desktop/Application/relate-data/dbmss/dbms-8c0ecfb9-233f-456f-bb53-715a986cb1ea",
                         "airportnew":"/Users/marcel/Library/Application Support/Neo4j Desktop/Application/relate-data/dbmss/dbms-786ac4bd-9d03-4fda-b510-9230a5f0f5fa",
                          "recommendations":"/Users/marcel/Library/Application Support/Neo4j Desktop/Application/relate-data/dbmss/dbms-e0a8a3a7-9923-42ba-bdc6-a54c7dc1f265",
                          "icijleaks":"/Users/marcel/Library/Application Support/Neo4j Desktop/Application/relate-data/dbmss/dbms-e93256e3-0282-4a59-84e6-7633fcd88179"}
    dict_databases_numbers={ #number of nodes, number of edges, avg number of properties nodes/edges
        "airports":[52944,136948,3.99,0.86],
        "airportnew":[52944,136948,4.39,2.13],
        "recommendations":[28863,166261,1.6,1.2],
        "icijleaks":[20165233,339267,1,0]
    }
    # validates and transform (scale) candidate indicators
    null_threshold = 0.5 #0.5
    distinct_low = 0.000001 #0.000001
    distinct_high = 0.96 #0.95
    correlation_threshold = 0.95 #0.95
    suffixes_for_removal=['_code','_id','_longitude','_latitude']

    # True = remove lines with at least one null value
    NONULLS = True

    #label
    #label=dict_databases_labels["airports"][0]

    for run in range(1):
        for db_name in dict_databases_labels.keys():
            print("database: ", db_name)

            stop_current_dbms()
            dbspec = DbSpec(db_name, dict_databases_homes[db_name], uri, user, dict_databases_passwords[db_name])
            start_dbms(dbspec)

            with Neo4jConnector(uri, user, db_name) as db:

                # finds relationship cardinalities
                print("Finding cardinalities")
                start_time = time.time()
                manyToOne, manyToMany = db.detect_relationship_cardinalities()
                end_time = time.time()
                timings_cardinalities = end_time - start_time

                for label in dict_databases_labels[db_name]:
                    print("Label: ",label)

                    start_time = time.time()

                    # get context and candidate indicators
                    # get * relationships for label
                    dfm2m = aggregate_m2m_properties_for_label(db.getDriver(), label, agg="sum", include_relationship_properties=True)

                    # get 1 relationships for label (and save those to csv)
                    out="sample_data/"+label+"_indicators.csv"
                    #do not send queries for edges if they do not have properties
                    if dict_databases_numbers[db_name][3] == float(0):
                        df121=db.fetch_as_dataframe(out,label,10,manyToOne,False)
                    else:
                        df121=db.fetch_as_dataframe(out,label,10,manyToOne,True)
                    #out join * and 1
                    dftemp = outer_join_features(df121, dfm2m, id_left="rootId", id_right="node_id", out_id="out1_id")

                    # get in degrees of label
                    dfdeg = in_degree_by_relationship_type(db.getDriver(),label)

                    # then outer join with dftemp
                    dffinal = outer_join_features(dftemp, dfdeg, id_left="out1_id", id_right="nodeId", out_id="out_id")

                    end_time = time.time()
                    indicatorsTimings = end_time - start_time


                    # validation
                    start_time = time.time()
                    # first remove unwanted indicators
                    dffinal,reportUW=drop_columns_by_suffix_with_report(dffinal,suffixes_for_removal)
                    # second remove correlated indicators
                    dffinal,reportCorr=remove_correlated_columns(dffinal,correlation_threshold)
                    # then check for variance and nulls, and scale
                    keep,report=process_dataframe(dffinal,null_threshold,distinct_low,distinct_high)

                    # union reportUW, reportCorr and report
                    report=pd.concat([report, reportUW, reportCorr], axis=0)

                    processedIndicators="sample_data/"+label+"_indicators_processed.csv"
                    processingReport="reports/"+label+"_indicators_processed.csv"

                    # if we remove lines with at least one null
                    if NONULLS:
                        keep=utility.remove_rows_with_nulls(keep)
                        processedIndicators = "sample_data/" + label + "_indicators_processed_nonulls.csv"

                    # contextualization
                    dist = schema_hops_from_label(db.getDriver(), label , include_relationship_types=True, directed=False)
                    keep = weight_df_by_schema_hops(keep, dist)

                    export(keep,report,processedIndicators,processingReport)


                    end_time = time.time()
                    validationTimings = end_time - start_time
                    #print('Completed in ', timings, 'seconds')

                    #call laplacian heuristics on data
                    start_time = time.time()
                    #partition=score(processedIndicators)
                    partition={}
                    end_time = time.time()
                    timingsPartition = end_time - start_time

                    #computes avg numerical properties for node type label
                    avgprop=db.getAvgPropByElem(label)[0]['avgNodeNumericProps']

                    #save to result dataframe
                    dfresults.loc[len(dfresults)] = [run, db_name, dict_databases_numbers[db_name][0], dict_databases_numbers[db_name][1], label, keep.shape[1] - 1, len(keep), avgprop, timings_cardinalities, indicatorsTimings, validationTimings, timingsPartition, partition]
                    #dfresults.to_csv('reports/tempres.csv', mode='a', header=True)
            stop_dbms(dbspec)
    dfresults.to_csv(fileResults, mode='a', header=True)
    # to analyze correlations in the result file
    #analyzeIndicatorDevisingTimes.main(fileResults,'report/correlations.csv')