from pathlib import Path

from neo4j import GraphDatabase, basic_auth
from typing import Any

import pandas as pd

import analyzeIndicatorDevisingTimes
import averageRunsCollectAndLatex
import plotPushdown
import utility
from experiments import heuristics
from many2many import aggregate_m2m_properties_for_label
from utility import outer_join_features
from validation import process_dataframe, export, remove_correlated_columns, drop_columns_by_suffix_with_report
from inDegrees import in_degree_by_relationship_type
import time
from orchestrate_neo4j import start_dbms, stop_dbms, DbSpec, stop_current_dbms

from laplacian_heuristics import score
from contextualization import schema_hops_from_label,weight_df_by_schema_hops

from typing import Iterable, List, Tuple, Optional, Dict, Set
import re


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

    def build_cypher_nodes(self, label: str, max_depth: Optional[int], many2one,suffixes,to_keep) -> str:
        """
        warning initially not on schema, was : all(rel...where apoc.node.degree.out(startNode(rel), apoc.rel.type(rel)) = 1)
        Build the Cypher query by inlining the label (labels can't be parameterized).
        Traverses only along steps where the current node has exactly one outgoing
        relationship of that type (functional chain).
        Keeps ONLY numeric properties via apoc.meta.cypher.type.
        """
        lbl = self.backtick_escape(label)
        depth = "" if max_depth is None else str(int(max_depth))
        props=to_keep

#       suffixRegex = '(' + '|'.join(map(re.escape, suffixes)) + r')$'
        suffixRegex = '.*(?:' + '|'.join(map(re.escape, suffixes)) + ')$'

        if to_keep==[]:
            # Note:
            # - Direction is OUTGOING (child -> parent). Flip to <-*0..- if your model is opposite.
            # - Properties are prefixed with the node's (first) label to avoid collisions.
            #AND NOT k =~ '{suffixRegex}' in second where
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

        else:
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
                                  [k IN {props}
                                     WHERE x[k] IS NOT NULL AND apoc.meta.cypher.type(x[k]) IN {NUMERIC_TYPES}
                                     | [head(labels(x)) + "_" + k, x[k]]
                                  ]
                                )
                             ] AS maps
                        RETURN id(n) AS rootId,
                               apoc.map.mergeList(maps) AS mergedNumericProperties
                        """

    def build_cypher_edges(self, label: str, max_depth: Optional[int],many2one,suffixes,to_keep) -> str:
        """
        warning, initially  not on schema, was : all(rel ... where apoc.node.degree.out(startNode(rel), apoc.rel.type(rel)) = 1)
        Build the Cypher query by inlining the label (labels can't be parameterized).
        Traverses only along steps where the current node has exactly one outgoing
        relationship of that type (functional chain).
        Keeps ONLY numeric properties via apoc.meta.cypher.type.
        """
        lbl = self.backtick_escape(label)
        depth = "" if max_depth is None else str(int(max_depth))
        props=to_keep

#        suffixRegex = '(' + '|'.join(map(re.escape, suffixes)) + r')$'
        suffixRegex = '.*(?:' + '|'.join(map(re.escape, suffixes)) + ')$'

        if to_keep==[]:
            # Note:
            # - Direction is OUTGOING (child -> parent). Flip to <-*0..- if your model is opposite.
            # - Properties are prefixed with the node's (first) label to avoid collisions.
            #AND NOT k =~ '{suffixRegex}'  in second where
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
        else:
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
                          [k IN {props}
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


    def fetch_as_dataframe(self, out, label: str, max_depth: Optional[int],many2one,checkedges=True, suffixes=[],to_keep=[]) -> pd.DataFrame:
        query=self.build_cypher_nodes(label, max_depth,many2one,suffixes,to_keep)
        result=self.execute_query(query)
        if checkedges:
            query_edge = self.build_cypher_edges(label, max_depth,many2one,suffixes,to_keep)
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



    def create_indexes_on_numeric_properties(self,
        numeric_types: Iterable[str] = ("Long", "Double", "Float", "Integer"),
        drop_suffixes: Optional[Iterable[str]] = None,
        include_nodes: bool = True,
        include_relationships: bool = True,
        dry_run: bool = False,
    ) -> Dict[str, List[Tuple[str, str]]]:
        """
        Create (or ensure) range indexes for all numeric properties across the graph,
        for both nodes and relationships (Neo4j 5+).

        Args:
            session: existing neo4j.Session (e.g., driver.session(database="neo4j"))
            numeric_types: APOC meta types considered numeric.
            drop_suffixes: suffixes to exclude (e.g., ["_id", "_code", "latitude"]).
            include_nodes: whether to create indexes on node properties.
            include_relationships: whether to create indexes on relationship properties.
            dry_run: if True, only return planned indexes; no CREATE is executed.

        Returns:
            {"node": [(label, property), ...], "relationship": [(rel_type, property), ...]}
            listing what was created (or would be created if dry_run=True).
        """

        session=self._driver.session()
        # Build one regex to drop unwanted suffixes (match at end of string).
        drop_re = ""
        if drop_suffixes:
            safe = [re.escape(s) for s in drop_suffixes]
            drop_re = r"(?:%s)$" % "|".join(safe)

        def _normalize_name(name: str) -> str:
            # Remove leading ':' if present and unwrap surrounding backticks
            s = name.lstrip(":")
            if len(s) >= 2 and s[0] == "`" and s[-1] == "`":
                s = s[1:-1]
            return s

        def qident(name: str) -> str:
            # Safe backtick-quoting for identifiers
            return "`" + name.replace("`", "``") + "`"

        created = {"node": [], "relationship": []}

        # -------- Nodes --------
        if include_nodes:
            meta_nodes = """
            CALL apoc.meta.nodeTypeProperties() 
            YIELD nodeType, propertyName, propertyTypes
            WITH nodeType, propertyName, propertyTypes
            WHERE any(t IN propertyTypes WHERE t IN $NUMERIC_TYPES)
              AND ($dropRe = '' OR NOT propertyName =~ $dropRe)
            RETURN DISTINCT nodeType AS rawLabel, propertyName AS property
            """
            node_rows = session.run(
                meta_nodes,
                NUMERIC_TYPES=list(numeric_types),
                dropRe=drop_re,
            )

            node_pairs = [(_normalize_name(r["rawLabel"]), r["property"]) for r in node_rows]

            for label, prop in node_pairs:
                cypher = f"CREATE INDEX IF NOT EXISTS FOR (n:{qident(label)}) ON (n.{qident(prop)})"
                if dry_run:
                    created["node"].append((label, prop))
                else:
                    session.run(cypher)
                    created["node"].append((label, prop))

        # -------- Relationships --------
        if include_relationships:
            meta_rels = """
            CALL apoc.meta.relTypeProperties()
            YIELD relType, propertyName, propertyTypes
            WITH relType, propertyName, propertyTypes
            WHERE any(t IN propertyTypes WHERE t IN $NUMERIC_TYPES)
              AND ($dropRe = '' OR NOT propertyName =~ $dropRe)
            RETURN DISTINCT relType AS rawType, propertyName AS property
            """
            rel_rows = session.run(
                meta_rels,
                NUMERIC_TYPES=list(numeric_types),
                dropRe=drop_re,
            )

            rel_pairs = [(_normalize_name(r["rawType"]), r["property"]) for r in rel_rows]

            for rtype, prop in rel_pairs:
                cypher = f"CREATE INDEX IF NOT EXISTS FOR ()-[r:{qident(rtype)}]-() ON (r.{qident(prop)})"
                if dry_run:
                    created["relationship"].append((rtype, prop))
                else:
                    session.run(cypher)
                    created["relationship"].append((rtype, prop))

        return created

    def numeric_properties_with_min_nonnull_ratio(
            self,
            min_nonnull_ratio: float,
            numeric_types: Iterable[str] = ("Long", "Double", "Float", "Integer"),
            drop_suffixes: Optional[Iterable[str]] = None,
    ) -> List[Dict]:
        """
        Return numeric properties on nodes *and* relationships with non-null ratio >= min_nonnull_ratio.
        Output rows: kind ('node'|'relationship'), label/relType, property, counts, ratio.
        """
        # End-anchored regex for suffix exclusion
        drop_re = ""
        if drop_suffixes:
            safe = [re.escape(s) for s in drop_suffixes]
            drop_re = r"(?:%s)$" % "|".join(safe)

        cypher = """
        // ===================== NODES =====================
        CALL apoc.meta.nodeTypeProperties()
        YIELD nodeType, propertyName, propertyTypes
        WITH
          // strip leading ':' if present
          CASE WHEN left(nodeType,1)=':' THEN substring(nodeType,1) ELSE nodeType END AS s1,
          propertyName AS property,
          propertyTypes
        WITH
          // strip surrounding backticks if present
          CASE
            WHEN size(s1)>=2 AND left(s1,1)='`' AND right(s1,1)='`'
              THEN substring(s1,1,size(s1)-2)
            ELSE s1
          END AS cleanLabel,
          property,
          propertyTypes
        WHERE any(t IN propertyTypes WHERE t IN $NUMERIC_TYPES)
          AND ($dropRe = '' OR NOT property =~ $dropRe)
        // count totals and non-nulls without dynamic pattern: use labels(n) and dynamic prop access
        MATCH (n)
        WHERE cleanLabel IN labels(n)
        WITH
          'node' AS kind,
          cleanLabel AS label,
          NULL AS relType,
          property,
          count(n) AS totalCount,
          sum(CASE WHEN n[property] IS NULL THEN 0 ELSE 1 END) AS nonNullCount
        WITH kind, label, relType, property, nonNullCount, totalCount,
             CASE WHEN totalCount = 0 THEN 0.0 ELSE toFloat(nonNullCount)/toFloat(totalCount) END AS nonNullRatio
        WHERE nonNullRatio >= $minNonNullRatio
        RETURN kind, label, relType, property, nonNullCount, totalCount, nonNullRatio

        UNION ALL

        // ===================== RELATIONSHIPS =====================
        CALL apoc.meta.relTypeProperties()
        YIELD relType, propertyName, propertyTypes
        WITH
          // remove surrounding backticks and any accidental leading ':'
            CASE WHEN left(relType,1)=':' THEN substring(relType,1) ELSE relType END AS sType,
          propertyName AS property,
          propertyTypes
        WITH
          CASE WHEN size(sType)>=2 AND left(sType,1)='`' AND right(sType,1)='`'
              THEN substring(sType,1,size(sType)-2)
            ELSE sType
          END AS cleanType,
          property,
          propertyTypes
        WHERE any(t IN propertyTypes WHERE t IN $NUMERIC_TYPES)
          AND ($dropRe = '' OR NOT property =~ $dropRe)
        // count totals and non-nulls without dynamic pattern: use type(r) and dynamic prop access
        MATCH ()-[r]-()
        WHERE type(r) = cleanType
        WITH
          'relationship' AS kind,
          NULL AS label,
          cleanType AS relType,
          property,
          count(r) AS totalCount,
          sum(CASE WHEN r[property] IS NULL THEN 0 ELSE 1 END) AS nonNullCount
        WITH kind, label, relType, property, nonNullCount, totalCount,
             CASE WHEN totalCount = 0 THEN 0.0 ELSE toFloat(nonNullCount)/toFloat(totalCount) END AS nonNullRatio
        WHERE nonNullRatio >= $minNonNullRatio
        RETURN kind, label, relType, property, nonNullCount, totalCount, nonNullRatio

        // consistent final ordering
        ORDER BY kind, nonNullRatio DESC, coalesce(label, relType), property
        """

        session = self._driver.session()
        recs = session.run(
            cypher,
            NUMERIC_TYPES=list(numeric_types),
            minNonNullRatio=float(min_nonnull_ratio),
            dropRe=drop_re,
        )
        #return [dict(r) for r in recs]
        return [r["property"] for r in recs]


    def numeric_properties_with_min_null_ratio(
            self,
            min_nonnull_ratio: float,
            numeric_types: Iterable[str] = ("Long", "Double", "Float", "Integer"),
            drop_suffixes: Optional[Iterable[str]] = None,
    ) -> List[Dict]:
        """
        Return numeric properties on nodes *and* relationships with non-null ratio >= min_nonnull_ratio.
        Output rows: kind ('node'|'relationship'), label/relType, property, counts, ratio.
        """
        # End-anchored regex for suffix exclusion
        drop_re = ""
        if drop_suffixes:
            safe = [re.escape(s) for s in drop_suffixes]
            drop_re = r"(?:%s)$" % "|".join(safe)

        cypher = """
        // ===================== NODES =====================
        CALL apoc.meta.nodeTypeProperties()
        YIELD nodeType, propertyName, propertyTypes
        WITH
          // strip leading ':' if present
          CASE WHEN left(nodeType,1)=':' THEN substring(nodeType,1) ELSE nodeType END AS s1,
          propertyName AS property,
          propertyTypes
        WITH
          // strip surrounding backticks if present
          CASE
            WHEN size(s1)>=2 AND left(s1,1)='`' AND right(s1,1)='`'
              THEN substring(s1,1,size(s1)-2)
            ELSE s1
          END AS cleanLabel,
          property,
          propertyTypes
        WHERE any(t IN propertyTypes WHERE t IN $NUMERIC_TYPES)
          AND ($dropRe = '' OR NOT property =~ $dropRe)
        // count totals and non-nulls without dynamic pattern: use labels(n) and dynamic prop access
        MATCH (n)
        WHERE cleanLabel IN labels(n)
        WITH
          'node' AS kind,
          cleanLabel AS label,
          NULL AS relType,
          property,
          count(n) AS totalCount,
          sum(CASE WHEN n[property] IS NULL THEN 0 ELSE 1 END) AS nonNullCount
        WITH kind, label, relType, property, nonNullCount, totalCount,
             CASE WHEN totalCount = 0 THEN 0.0 ELSE toFloat(nonNullCount)/toFloat(totalCount) END AS nonNullRatio
        WHERE nonNullRatio < $minNonNullRatio
        RETURN kind, label, relType, property, nonNullCount, totalCount, nonNullRatio

        UNION ALL

        // ===================== RELATIONSHIPS =====================
        CALL apoc.meta.relTypeProperties()
        YIELD relType, propertyName, propertyTypes
        WITH
          // remove surrounding backticks and any accidental leading ':'
            CASE WHEN left(relType,1)=':' THEN substring(relType,1) ELSE relType END AS sType,
          propertyName AS property,
          propertyTypes
        WITH
          CASE WHEN size(sType)>=2 AND left(sType,1)='`' AND right(sType,1)='`'
              THEN substring(sType,1,size(sType)-2)
            ELSE sType
          END AS cleanType,
          property,
          propertyTypes
        WHERE any(t IN propertyTypes WHERE t IN $NUMERIC_TYPES)
          AND ($dropRe = '' OR NOT property =~ $dropRe)
        // count totals and non-nulls without dynamic pattern: use type(r) and dynamic prop access
        MATCH ()-[r]-()
        WHERE type(r) = cleanType
        WITH
          'relationship' AS kind,
          NULL AS label,
          cleanType AS relType,
          property,
          count(r) AS totalCount,
          sum(CASE WHEN r[property] IS NULL THEN 0 ELSE 1 END) AS nonNullCount
        WITH kind, label, relType, property, nonNullCount, totalCount,
             CASE WHEN totalCount = 0 THEN 0.0 ELSE toFloat(nonNullCount)/toFloat(totalCount) END AS nonNullRatio
        WHERE nonNullRatio < $minNonNullRatio
        RETURN kind, label, relType, property, nonNullCount, totalCount, nonNullRatio

        // consistent final ordering
        ORDER BY kind, nonNullRatio DESC, coalesce(label, relType), property
        """

        session = self._driver.session()
        recs = session.run(
            cypher,
            NUMERIC_TYPES=list(numeric_types),
            minNonNullRatio=float(min_nonnull_ratio),
            dropRe=drop_re,
        )
        #return [dict(r) for r in recs]
        return [r["property"] for r in recs]



    def drop_indexes_on_numeric_properties(
            self,
            numeric_types: Iterable[str] = ("Long", "Double", "Float", "Integer"),
            drop_suffixes: Optional[Iterable[str]] = None,
            include_nodes: bool = True,
            include_relationships: bool = True,
            dry_run: bool = False,
    ) -> Dict[str, List[str]]:
        """
        Drop (if they exist) all Neo4j property indexes that index only numeric properties,
        for nodes and/or relationships.

        - Uses APOC store metadata to decide what is numeric (no graph scan).
        - Skips indexes owned by constraints.
        - Only drops RANGE/BTREE property indexes (not FULLTEXT/LOOKUP/TEXT/POINT).

        Returns:
            {"node": [index_name, ...], "relationship": [index_name, ...]}
            listing the indexes that were (or would be, in dry_run) dropped.
        """
        # Build optional end-anchored regex to exclude suffixes from "numeric" set
        drop_re = ""
        if drop_suffixes:
            drop_re = r"(?:%s)$" % "|".join(re.escape(s) for s in drop_suffixes)

        def qident(name: str) -> str:
            return "`" + name.replace("`", "``") + "`"

        # --- 1) Collect numeric properties per node label / rel type ---
        numeric_node_props: Dict[str, Set[str]] = {}
        numeric_rel_props: Dict[str, Set[str]] = {}

        session=self._driver.session()
        if include_nodes:
            rows = session.run(
                """
                CALL apoc.meta.nodeTypeProperties()
                YIELD nodeType, propertyName, propertyTypes
                WITH
                  CASE WHEN left(nodeType,1)=':' THEN substring(nodeType,1) ELSE nodeType END AS s1,
                  propertyName AS property,
                  propertyTypes
                WITH
                  CASE WHEN size(s1)>=2 AND left(s1,1)='`' AND right(s1,1)='`'
                       THEN substring(s1,1,size(s1)-2) ELSE s1 END AS label,
                  property, propertyTypes
                WHERE any(t IN propertyTypes WHERE t IN $NUMERIC_TYPES)
                  AND ($dropRe = '' OR NOT property =~ $dropRe)
                RETURN label, property
                """,
                NUMERIC_TYPES=list(numeric_types),
                dropRe=drop_re,
            )
            for r in rows:
                numeric_node_props.setdefault(r["label"], set()).add(r["property"])

        if include_relationships:
            rows = session.run(
                """
                CALL apoc.meta.relTypeProperties()
                YIELD relType, propertyName, propertyTypes
                WITH
                  CASE WHEN left(relType,1)=':' THEN substring(relType,1) ELSE relType END AS s1,
                  propertyName AS property,
                  propertyTypes
                WITH
                  CASE WHEN size(s1)>=2 AND left(s1,1)='`' AND right(s1,1)='`'
                       THEN substring(s1,1,size(s1)-2) ELSE s1 END AS rtype,
                  property, propertyTypes
                WHERE any(t IN propertyTypes WHERE t IN $NUMERIC_TYPES)
                  AND ($dropRe = '' OR NOT property =~ $dropRe)
                RETURN rtype, property
                """,
                NUMERIC_TYPES=list(numeric_types),
                dropRe=drop_re,
            )
            for r in rows:
                numeric_rel_props.setdefault(r["rtype"], set()).add(r["property"])

        # Quick outs if nothing numeric found
        if not numeric_node_props and not numeric_rel_props:
            return {"node": [], "relationship": []}

        # --- 2) List existing indexes ---
        index_rows = session.run(
            """
            SHOW INDEXES YIELD name, type, entityType, labelsOrTypes, properties, owningConstraint
            RETURN name, type, entityType, labelsOrTypes, properties, owningConstraint
            """
        )

        droppable_nodes: List[str] = []
        droppable_rels: List[str] = []

        ALLOWED_TYPES = {"RANGE", "BTREE"}  # property indexes we may drop

        for r in index_rows:
            name = r["name"]
            idx_type = r["type"]  # e.g., RANGE, FULLTEXT, LOOKUP, POINT, TEXT
            entity = r["entityType"]  # NODE or RELATIONSHIP
            labels_or_types = r["labelsOrTypes"] or []
            props = r["properties"] or []
            owning = r["owningConstraint"]  # non-null => tied to a constraint

            # Skip constraint-owned or non-supported types
            if owning is not None:
                continue
            if idx_type not in ALLOWED_TYPES:
                continue
            if not labels_or_types or not props:
                continue

            # We only handle standard property indexes (single label/type)
            label_or_type = labels_or_types[0]

            if entity == "NODE" and include_nodes:
                numeric_set = numeric_node_props.get(label_or_type, set())
                # drop only if ALL properties in the index are numeric for this label
                if numeric_set and all(p in numeric_set for p in props):
                    droppable_nodes.append(name)

            elif entity == "RELATIONSHIP" and include_relationships:
                numeric_set = numeric_rel_props.get(label_or_type, set())
                if numeric_set and all(p in numeric_set for p in props):
                    droppable_rels.append(name)

        # --- 3) Drop them (or dry-run) ---
        dropped = {"node": [], "relationship": []}

        for name in droppable_nodes:
            if dry_run:
                dropped["node"].append(name)
            else:
                session.run(f"DROP INDEX {qident(name)} IF EXISTS")
                dropped["node"].append(name)

        for name in droppable_rels:
            if dry_run:
                dropped["relationship"].append(name)
            else:
                session.run(f"DROP INDEX {qident(name)} IF EXISTS")
                dropped["relationship"].append(name)

        return dropped




    def count_numeric_properties(self):
        """
        Count the total number of numeric properties in all nodes and relationships
        of a Neo4j database.

        Parameters
        ----------
        uri : str
            Bolt URI of the database, e.g., "bolt://localhost:7687"
        user : str
            Username for authentication
        password : str
            Password for authentication
        db : str
            Database name (default: "neo4j")

        Returns
        -------
        dict
            {
              "nodes_numeric_props": int,
              "rels_numeric_props": int,
              "total_numeric_props": int
            }
        """

        with self._driver.session() as session:
            # Cypher list of numeric property types
            numeric_types = ["INTEGER", "FLOAT", "DOUBLE", "LONG"]

            # --- Count numeric properties in nodes ---
            cypher = """
            CALL {
                MATCH (n)
                UNWIND keys(n) AS k
                WITH k, apoc.meta.cypher.type(n[k]) AS t
                WHERE t IN $numeric_types
                RETURN collect(DISTINCT k) AS node_names
            }
            CALL {
                MATCH ()-[r]->()
                UNWIND keys(r) AS k
                WITH k, apoc.meta.cypher.type(r[k]) AS t
                WHERE t IN $numeric_types
                RETURN collect(DISTINCT k) AS rel_names
            }
            RETURN node_names, rel_names, size(node_names) AS node_count, size(rel_names) AS rel_count,
                size(apoc.coll.toSet(node_names + rel_names)) AS total   
            """
            rec =  session.run(cypher, numeric_types=numeric_types).single()
            node_names = sorted(rec["node_names"])
            rel_names = sorted(rec["rel_names"])
            node_count = rec["node_count"]
            rel_count = rec["rel_count"]
            total = rec["total"]

        return node_count + rel_count


def main(pushdown,null_ratio,nbRuns):
    current_time = time.localtime()
    formatted_time = time.strftime("%d-%m-%y:%H:%M:%S", current_time)
    fileResults = 'reports/results_' + formatted_time + '.csv'
    column_names = ['run','pushdown','database', 'N','E', 'label', 'indicators#', 'nodes#', 'avgLabelProp', 'time_Preprocessing', 'time_Cardinalities', 'time_Indicators','time_Validation','time_total','ratio_prop_dropped']
    dfresults = pd.DataFrame(columns=column_names)

    #  URI/user/password
    uri="bolt://localhost:7687"
    user="neo4j"
    dict_databases_labels={#"airports":["Airport","Country","City"],
                           #"airportnew": ["Airport", "Country", "City"],
                           "airportnew": ["Airport", "City"], #
                        # "recommendations":["Actor","Movie","Director"],
                           #"icijleaks":["Entity", "Intermediary", "Officer"]
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
    # thresholds for validating and transforming (scale,contextualize) candidate indicators
    null_threshold = null_ratio
    distinct_low = 0.000001
    distinct_high = 1
    correlation_threshold = 0.98
    unwanted_suffixes=['_code','_id','longitude','latitude'] # for unwanted properties

    # True = remove lines with at least one null value
    NONULLS = True

    # if unwanted properties, acceptable density (validation) are pushed down indicator collection
    PUSHDOWN = pushdown

    # should we drop/create all indices on numerical properties
    DROP_INDEX = True
    CREATE_INDEX = True

    # for tests
    nbRuns=nbRuns

    for run in range(nbRuns):
        for db_name in dict_databases_labels.keys():
            print("database: ", db_name)

            stop_current_dbms()
            dbspec = DbSpec(db_name, dict_databases_homes[db_name], uri, user, dict_databases_passwords[db_name])
            start_dbms(dbspec)

            with Neo4jConnector(uri, user, db_name) as db:
                #counts the number of numeric properties
                resprop=db.count_numeric_properties()
                # drop index
                if DROP_INDEX:
                    res = db.drop_indexes_on_numeric_properties(
                        numeric_types=("Long", "Double", "Float", "Integer"),
                        drop_suffixes=[],  # optional
                        include_nodes=True,
                        include_relationships=True,
                        dry_run=False,  # set True to preview
                    )
                # create indices
                if CREATE_INDEX:
                    print("Indexing numerical properties")
                    start_time = time.time()
                    res = db.create_indexes_on_numeric_properties(
                        numeric_types=("Long", "Double", "Float", "Integer"),
                        drop_suffixes=unwanted_suffixes,
                        include_nodes=True,
                        include_relationships=True,
                        dry_run=False,  # set True to preview without creating
                    )
                    end_time = time.time()
                    timings_indexes = end_time - start_time
                    print("Node indexes:", res["node"])
                    print("Relationship indexes:", res["relationship"])
                else:
                    timings_indexes = 0

                # finds relationship cardinalities
                print("Finding cardinalities")
                start_time = time.time()
                manyToOne, manyToMany = db.detect_relationship_cardinalities()
                end_time = time.time()
                timings_cardinalities = end_time - start_time

                if PUSHDOWN:
                    # finding properties with acceptable density
                    start_time = time.time()
                    props = db.numeric_properties_with_min_nonnull_ratio(
                        min_nonnull_ratio=1-null_threshold,
                        numeric_types=("Long", "Double", "Float", "Integer"),
                        drop_suffixes=unwanted_suffixes,  # optional
                    )
                    # print('time nulls:', time.time() - start_time)
                    #suffixes_for_removal = unwanted_suffixes + props
                    suffixes_for_removal=[]
                    #to_keep=props
                    to_keep=[s for s in props if not any(s.endswith(sfx) for sfx in unwanted_suffixes)]
                    print("number of properties to keep:", len(to_keep))
                    ratio_dropped= 100*(resprop - len(to_keep)) /resprop
                    end_time = time.time()
                    timings_density = end_time - start_time
                else:
                    props = db.numeric_properties_with_min_nonnull_ratio(
                        min_nonnull_ratio=1 - null_threshold,
                        numeric_types=("Long", "Double", "Float", "Integer"),
                        drop_suffixes=unwanted_suffixes,  # optional
                    )
                    to_keep = [s for s in props if not any(s.endswith(sfx) for sfx in unwanted_suffixes)]
                    ratio_dropped = 100 * (resprop - len(to_keep)) / resprop
                    toPassForValidation = db.numeric_properties_with_min_null_ratio(
                        min_nonnull_ratio=1 - null_threshold,
                        numeric_types=("Long", "Double", "Float", "Integer"),
                        drop_suffixes=unwanted_suffixes,  # optional
                    )
                    #ratio_dropped = 100*len(toPassForValidation) / resprop
                    suffixes_for_removal = []
                    to_keep = [] #no pushdown
                    timings_density = 0
                timings_preprocessing=timings_indexes+timings_density

                for label in dict_databases_labels[db_name]:
                    print("Label: ",label)

                    # collect candidate indicators
                    print("Collecting candidate indicators")
                    start_time = time.time()

                    # get context
                    # get * relationships properties for label
                    dfm2m = aggregate_m2m_properties_for_label(db.getDriver(), label, agg="sum", include_relationship_properties=True,only_reltypes=manyToMany,suffixes=suffixes_for_removal,to_keep=to_keep)
                    #print('* props:', time.time() - start_time)

                    # get 1 relationships properties for label (and save those to csv)
                    out="sample_data/"+label+"_indicators.csv"
                    #do not send queries for edges if they do not have properties
                    if dict_databases_numbers[db_name][3] == float(0):
                        df121=db.fetch_as_dataframe(out,label,5,manyToOne,False, suffixes_for_removal,to_keep=to_keep)
                    else:
                        df121=db.fetch_as_dataframe(out,label,5,manyToOne,True, suffixes_for_removal,to_keep=to_keep)
                    #print('1 props:', time.time() - start_time)

                    #out join * and 1
                    dftemp = outer_join_features(df121, dfm2m, id_left="rootId", id_right="node_id", out_id="out1_id")

                    # get in degrees of label
                    dfdeg = in_degree_by_relationship_type(db.getDriver(),label)
                    #print('degrees:', time.time() - start_time)

                    # then outer join with dftemp
                    dffinal = outer_join_features(dftemp, dfdeg, id_left="out1_id", id_right="nodeId", out_id="outid")

                    end_time = time.time()
                    indicatorsTimings = end_time - start_time


                    # validation
                    print("Validating candidate indicators")

                    if PUSHDOWN:
                        suffixes_unwanted=[]
                    else:
                        suffixes_unwanted=unwanted_suffixes+toPassForValidation

                    start_time = time.time()
                    # first remove unwanted indicators
                    if not PUSHDOWN:
                        dffinal,reportUW=drop_columns_by_suffix_with_report(dffinal,suffixes_unwanted)
                    # second remove correlated indicators
                    dffinal,reportCorr=remove_correlated_columns(dffinal,correlation_threshold)
                    # then check for variance and nulls, and scale
                    keep,report=process_dataframe(dffinal,null_threshold,distinct_low,distinct_high,PUSHDOWN)

                    # union reportUW, reportCorr and report
                    if not PUSHDOWN:
                        report=pd.concat([report, reportUW, reportCorr], axis=0)
                    else:
                        report = pd.concat([report, reportCorr], axis=0)

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

                    #call laplacian heuristics on data
                    #start_time = time.time()
                    #partition=score(processedIndicators)
                    #partition={}
                    #end_time = time.time()
                    #timingsPartition = end_time - start_time
                    timings_total=indicatorsTimings+validationTimings +timings_preprocessing+timings_cardinalities

                    #computes avg numerical properties for node type label
                    avgprop=db.getAvgPropByElem(label)[0]['avgNodeNumericProps']

                    #save to result dataframe
                    dfresults.loc[len(dfresults)] = [run, PUSHDOWN, db_name, dict_databases_numbers[db_name][0], dict_databases_numbers[db_name][1], label, keep.shape[1] - 1, len(keep), avgprop, timings_preprocessing, timings_cardinalities, indicatorsTimings, validationTimings, timings_total, ratio_dropped ]
                    if nbRuns!=1:
                        dfresults.to_csv('reports/tempres'+ formatted_time +'.csv', mode='a', header=True)
            stop_dbms(dbspec)
    dfresults.to_csv(fileResults, mode='a', header=True)
    # to average results by labels
    out,latex=averageRunsCollectAndLatex.average_time_columns_by_label_to_latex_pretty(csv_path=fileResults,label_col="label",float_precision=1,output_csv="reports/averaged_time_by_label.csv", output_tex="reports/averaged_time_by_label.tex",)
    print("\n===== LaTeX Preview =====\n")
    print(latex)
    # to analyze correlations in the result file
    analyzeIndicatorDevisingTimes.main(Path('reports/averaged_time_by_label.csv'),Path('reports'))
    #return  db_name, label, keep.shape[1] - 1, len(keep), timings_preprocessing, timings_cardinalities, indicatorsTimings, validationTimings, timings_total, ratio_dropped
    return dfresults

def testPushdown():
    fileResults = 'reports/results_test_pushdown.csv'
    #column_names = ['run', 'pushdown', 'null_ratio', 'database', 'label', 'indicators#', 'nodes#',
    #                'time_Preprocessing', 'time_Cardinalities', 'time_Indicators', 'time_Validation', 'time_total',
    #                'ratio_prop_dropped']
    column_names = ['run','pushdown','database', 'N','E', 'label', 'indicators#', 'nodes#', 'avgLabelProp', 'time_Preprocessing', 'time_Cardinalities', 'time_Indicators','time_Validation','time_total','ratio_prop_dropped']
    dfresults = pd.DataFrame(columns=column_names)
    for run in range(3):
        for pushdown in [False,True]:
            #for null_ratio in [0.5, 0.3, 0.26, 0.25, 0.1]:
            for null_ratio in [0.5, 0.3]:
                #db_name, label, indicators, nodes, timings_preprocessing, timings_cardinalities, indicatorsTimings,validationTimings, timings_total, ratio_dropped=main(pushdown,null_ratio,1)
                #dfresults.loc[len(dfresults)] = [run, pushdown, null_ratio ,  db_name, label, indicators, nodes, timings_preprocessing, timings_cardinalities, indicatorsTimings,validationTimings, timings_total, ratio_dropped]
                result = main(pushdown, null_ratio, 1)
                #dfresults.loc[len(dfresults)] =
                dfresults=pd.concat([dfresults,result])
    dfresults.to_csv(fileResults, mode='a', header=True)
    plotPushdown.main(fileResults)


if __name__ == "__main__":
    testPushdown()
    #main(True, 0.5, 1)

