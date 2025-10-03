import itertools
import time
from collections import defaultdict
from collections.abc import Generator
from statistics import pvariance, variance
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import utility
from contextualization import schema_hops_from_label, weight_df_by_schema_hops
from inDegrees import in_degree_by_relationship_type
from laplacian_heuristics import score
from many2many import aggregate_m2m_properties_for_label
from neo4j import Driver, GraphDatabase, basic_auth
from orchestrate_neo4j import DbSpec, start_dbms, stop_current_dbms, stop_dbms
from scipy.stats import variation
from utility import outer_join_features
from validation import export, process_dataframe, remove_correlated_columns

NUMERIC_TYPES = [
    # APOC meta cypher type names (Neo4j 4/5)
    "INTEGER",
    "FLOAT",
    "Number",
    "Long",
    "Double",
    # (If you want numeric lists too, add: "LIST OF INTEGER", "LIST OF FLOAT", etc.)
]


class Neo4jConnector:
    """A simple Neo4j connection manager and query runner."""

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        encrypted: bool = False,
        **driver_kwargs: Any,
    ) -> None:
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
            **driver_kwargs,
        )

    def get_driver(self) -> Driver:
        return self._driver

    def close(self) -> None:
        """Close the underlying driver connection."""
        self._driver.close()

    def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Run a Cypher query and return a list of records as dictionaries.

        :param query: Cypher query string
        :param parameters: Query parameters dict
        :param database: Optional database name (for multi-db setups)
        :return: List of record dictionaries
        """
        with self._driver.session(database=database) as session:
            result = session.run(query, parameters or {})
            for record in result:
                yield record.data()

    def __enter__(self) -> "Neo4jConnector":
        """Enable use as a context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Ensure the driver is closed on exit."""
        self.close()

    def get_numerical_properties(self, label: str) -> Set[str]:
        """Return the names of numerical properties of nodes with label."""
        properties = self.execute_query(
            """
            CALL apoc.meta.nodeTypeProperties({includeLabels:$labels})
            YIELD propertyName, propertyTypes
            WHERE ANY(t IN propertyTypes WHERE t IN $num_types)
            RETURN DISTINCT propertyName AS prop;
            """,
            parameters={
                "labels": [label],
                "num_types": NUMERIC_TYPES,
            },
        )
        return {p["prop"] for p in properties}

    def variance_of_property(self, property: str, label: str):
        # get variance indicators of property for nodes of label
        # returns nb values, nb distinct values, variance of sample, variance of population, coefficient of variation
        # first need all the values
        values = self.execute_query(f"MATCH (n:{label}) RETURN n.{property} AS prop")
        tab_val = [v['prop'] for v in values if v['prop'] is not None]

        return (
            tab_val,
            len(tab_val),
            len(set(tab_val)),
            variance(tab_val),
            pvariance(tab_val),
            variation(tab_val),
        )

    def get_valid_properties(self, label: str) -> Set[str]:
        result = set()
        properties = self.get_numerical_properties(label)
        for prop in properties:
            tab, nb, nbd, v, pv, cv = self.variance_of_property(prop, label)
            if 1 < nbd < nb:  # fixme: check better ways of validating
                result.add(prop)
        return result

    def get_relation_cardinality(self) -> Dict[str, Set[str]]:
        cardinalities = self.execute_query("""
        MATCH (x)-[r]->(y)
        WITH
            type(r) AS name, head(labels(x)) AS label_x, head(labels(y)) AS label_y,
            count(distinct x) AS count_x, count(distinct y) AS count_y, count(distinct r) AS count_r
        RETURN name,
            CASE WHEN count_y < count_r THEN label_x ELSE label_y END AS source,
            CASE WHEN count_y < count_r THEN label_y ELSE label_x END AS target,
            count_x < count_r AND count_y < count_r AS is_many_to_many
        """)

        dict_rel = {"manyToMany": set(), "oneToMany": set()}
        for card in cardinalities:
            if card["is_many_to_many"]:
                dict_rel["manyToMany"].add(card["name"])
            else:
                dict_rel["oneToMany"].add(card["name"])

        return dict_rel

    def create_dataset_for_label(self, label: str) -> pd.DataFrame:
        properties = self.get_valid_properties(label)
        properties_str = " ".join(f", n.{prop} AS {prop}" for prop in properties)
        values = self.execute_query(f"MATCH (n:{label}) RETURN elementId(n)" + properties_str)
        return pd.DataFrame.from_records(values)

    def get_degree_of_relation_for_label(self, label: str) -> Tuple[Set[str], Dict[str, Dict[str, int]]]:
        # returns degree of type for nodeId
        # tabProp=self.getValidProperties(label)
        relation_types = {rel["relationshipType"] for rel in self.execute_query("call db.relationshipTypes()")}
        print(relation_types)

        dict_rel = defaultdict(dict)
        for rel in relation_types:
            degrees = self.execute_query(f"MATCH (n:{label})-[r:{rel}]-() RETURN elementId(n), count(r) AS count")
            # print(degrees)
            for d in degrees:
                dict_rel[d["elementId(n)"]][rel] = d["count"]

        print(dict_rel)
        relation_types.add("elementId")
        return relation_types, dict_rel

    def get_best_page_rank(self, label: str, relationship: str):
        # retrieve the id of the node with highest page rank value
        # first remove graph if exists
        self.execute_query("CALL gds.graph.drop('myGraph', false) YIELD graphName;")
        self.execute_query(f"""
        MATCH (source:{label})-[:{relationship}]->(target:{label})
        RETURN gds.graph.project('myGraph',  source,  target);
        """)
        result = self.execute_query("""
        CALL gds.pageRank.stream('myGraph')
        YIELD nodeId, score RETURN elementId(gds.util.asNode(nodeId)) AS elementId, score
        ORDER BY score DESC, elementId ASC limit 1
        """)
        return next(result)["elementId"]

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
        # return result
        return (
            [rec["relType"] for rec in result if rec["cardinality"] == "many-to-one"],
            [rec["relType"] for rec in result if rec["cardinality"] == "many-to-many"],
        )

    def build_cypher_nodes(self, label: str, max_depth: Optional[int], many2one) -> str:
        """
        Warning initially not on schema, was : all(rel...where apoc.node.degree.out(startNode(rel), apoc.rel.type(rel)) = 1)
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
          all(rel IN relationships(p) WHERE type(rel) in {many2one})
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

    def build_cypher_edges(self, label: str, max_depth: Optional[int], many2one) -> str:
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
          all(rel IN relationships(p) WHERE type(rel) in {many2one})
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

    def fetch_as_dataframe(
        self, out, label: str, max_depth: Optional[int], many2one, checkedges: bool = True
    ) -> pd.DataFrame:
        query = self.build_cypher_nodes(label, max_depth, many2one)
        result = self.execute_query(query)

        if checkedges:
            query_edge = self.build_cypher_edges(label, max_depth, many2one)
            result_edges = self.execute_query(query_edge)
            result = itertools.chain(result_edges)

        rows = ({"rootId": rec["rootId"], **rec.get("mergedNumericProperties", {})} for rec in result)

        df = pd.DataFrame.from_records(rows).sort_values("rootId").reset_index(drop=True)
        df.to_csv(out, index=False)
        print(f"Saved {len(df)} rows to: {out}")
        print("Columns:", ", ".join(df.columns))
        return df

    def get_avg_prop_by_elem(self, label: str) -> Generator[Dict[str, float], None, None]:
        query = f"""
        CALL {{
            MATCH (n:{label})
            WITH n, [k IN keys(n) WHERE apoc.meta.cypher.type(n[k]) IN $num_types] AS numericProps
            RETURN avg(size(numericProps)) AS avgNodeNumericProps
        }}
        CALL {{
            MATCH ()-[r]-()
            WITH r, [k IN keys(r) WHERE apoc.meta.cypher.type(r[k]) IN $num_types] AS numericProps
            RETURN avg(size(numericProps)) AS avgRelNumericProps
        }}
        RETURN avgNodeNumericProps, avgRelNumericProps
        """
        return self.execute_query(query, parameters={"num_types": NUMERIC_TYPES})

    def index_numeric_properties(self, label: str) -> List[Dict[str, object]]:
        """
        Create range indexes for all numeric (Integer/Float) properties
        of nodes with the given label.

        Returns
        -------
        List[Dict[str, object]]
            One dict per property with keys:
            - property: str         # property name
            - index_name: str       # created or reused index name
            - created: bool         # True if a new index was created

        """
        # 1) Discover numeric properties for the label using built-in metadata
        discover_props_cypher = """
        CALL db.schema.nodeTypeProperties() 
        YIELD nodeLabels, propertyName, propertyTypes
        WHERE $label IN nodeLabels 
          AND any(t IN propertyTypes WHERE t IN ['Integer','Float'])
        RETURN collect(DISTINCT propertyName) AS props
        """

        # 2) Create (if not exists) a range index for each numeric property
        results: List[Dict[str, object]] = []
        with self.driver.session() as session:
            rec = session.run(discover_props_cypher, {"label": label}).single()
            props = rec["props"] if rec and rec.get("props") else []

            for prop in props:
                # Safe names; quote label/prop in Cypher; index name is optional but helpful
                idx_name = f"idx_{label}_{prop}".replace("`", "").lower()
                create_idx_cypher = f"""
                CREATE INDEX {idx_name} IF NOT EXISTS
                FOR (n:`{label}`) ON (n.`{prop}`)
                """
                summary = session.run(create_idx_cypher).consume()
                created = summary.counters.indexes_added > 0
                results.append({"property": prop, "index_name": idx_name, "created": created})

        return results


# Example usage:
if __name__ == "__main__":
    current_time = time.localtime()
    formatted_time = time.strftime("%d-%m-%y:%H:%M:%S", current_time)
    fileResults = "reports/results_" + formatted_time + ".csv"
    column_names = [
        "database",
        "N",
        "E",
        "label",
        "indicators#",
        "nodes#",
        "avgLabelProp",
        "time_Cardinalities",
        "time_Indicators",
        "time_Validation",
        "time_Partition",
        "partition",
    ]
    dfresults = pd.DataFrame(columns=column_names)

    #  URI/user/password
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "airports"
    tab_databases = ["airports", "icijleaks", "recommendations"]
    dict_databases_labels = {
        "airports": ["Airport", "Country", "City"],
        "recommendations": ["Actor", "Movie", "Director"],
        # ,"icijleaks":["Entity", "Intermediary", "Officer"]
        "icijleaks": ["Intermediary"],
    }
    dict_databases_homes = {
        "airports": "/Users/marcel/Library/Application Support/Neo4j Desktop/Application/relate-data/dbmss/dbms-8c0ecfb9-233f-456f-bb53-715a986cb1ea",
        "recommendations": "/Users/marcel/Library/Application Support/Neo4j Desktop/Application/relate-data/dbmss/dbms-e0a8a3a7-9923-42ba-bdc6-a54c7dc1f265",
        "icijleaks": "/Users/marcel/Library/Application Support/Neo4j Desktop/Application/relate-data/dbmss/dbms-e93256e3-0282-4a59-84e6-7633fcd88179",
    }
    dict_databases_numbers = {  # number of nodes, number of edges, avg number of properties nodes/edges
        "airports": [52944, 136948, 3.99, 0.86],
        "recommendations": [28863, 166261, 1.6, 1.2],
        "icijleaks": [20165233, 339267, 1, 0],
    }
    # validates and transform (scale) candidate indicators
    null_threshold = 0.5  # 0.5
    distinct_low = 0.000001  # 0.000001
    distinct_high = 0.95  # 0.95
    correlation_threshold = 0.95  # 0.95

    # True = remove lines with at least one null value
    NONULLS = True

    # label
    # label=dict_databases_labels["airports"][0]

    for password in dict_databases_labels:
        print("database: ", password)

        stop_current_dbms()
        dbspec = DbSpec(password, dict_databases_homes[password], uri, user, password)
        start_dbms(dbspec)

        with Neo4jConnector(uri, user, password) as db:
            # finds relationship cardinalities
            print("Finding cardinalities")
            start_time = time.time()
            manyToOne, manyToMany = db.detect_relationship_cardinalities()
            end_time = time.time()
            timings_cardinalities = end_time - start_time

            for label in dict_databases_labels[password]:
                print("Label: ", label)

                start_time = time.time()

                # get context and candidate indicators
                # get * relationships for label
                dfm2m = aggregate_m2m_properties_for_label(
                    db.get_driver(),
                    label,
                    agg="sum",
                    include_relationship_properties=True,
                )

                # get 1 relationships for label (and save those to csv)
                out = "sample_data/" + label + "_indicators.csv"
                # do not send queries for edges if they do not have properties
                df121 = db.fetch_as_dataframe(out, label, 10, manyToOne, dict_databases_numbers[password][3] != 0)
                # out join * and 1
                dftemp = outer_join_features(df121, dfm2m, id_left="rootId", id_right="node_id", out_id="out1_id")

                # get in degrees of label
                dfdeg = in_degree_by_relationship_type(db.get_driver(), label)

                # then outer join with dftemp
                dffinal = outer_join_features(dftemp, dfdeg, id_left="out1_id", id_right="nodeId", out_id="out_id")

                end_time = time.time()
                indicatorsTimings = end_time - start_time

                # validation
                start_time = time.time()
                # first remove correlated columns
                dffinal = remove_correlated_columns(dffinal, correlation_threshold)
                # then check for variance and nulls, and scale
                keep, report = process_dataframe(dffinal, null_threshold, distinct_low, distinct_high)

                processedIndicators = f"sample_data/{label}_indicators_processed.csv"
                processingReport = f"reports/{label}_indicators_processed.csv"

                # if we remove lines with at least one null
                if NONULLS:
                    keep = utility.remove_rows_with_nulls(keep)
                    processedIndicators = f"sample_data/{label}_indicators_processed_nonulls.csv"

                # contextualization
                dist = schema_hops_from_label(
                    db.get_driver(),
                    label,
                    include_relationship_types=True,
                    directed=False,
                )
                keep = weight_df_by_schema_hops(keep, dist)

                export(keep, report, processedIndicators, processingReport)

                end_time = time.time()
                validationTimings = end_time - start_time
                # print('Completed in ', timings, 'seconds')

                # call laplacian heuristics on data
                start_time = time.time()
                partition = score(processedIndicators)
                end_time = time.time()
                timingsPartition = end_time - start_time

                # computes avg numerical properties for node type label
                avgprop = next(db.get_avg_prop_by_elem(label))["avgNodeNumericProps"]

                # save to result dataframe
                dfresults.loc[len(dfresults)] = [
                    password,
                    dict_databases_numbers[password][0],
                    dict_databases_numbers[password][1],
                    label,
                    keep.shape[1] - 1,
                    len(keep),
                    avgprop,
                    timings_cardinalities,
                    indicatorsTimings,
                    validationTimings,
                    timingsPartition,
                    partition,
                ]

        stop_dbms(dbspec)
    dfresults.to_csv(fileResults, mode="a", header=True)
