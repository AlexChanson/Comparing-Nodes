import re
from typing import Iterable, List, Dict, Optional, Literal
from neo4j import Driver, Session
import pandas as pd

AggFn = Literal["sum", "avg", "min", "max", "count"]

def _backtick(label: str) -> str:
    # Safely quote labels that may contain special chars/spaces
    return f"`{label.replace('`', '``')}`"

def find_many_to_many_reltypes(session: Session, label: str) -> List[str]:
    """
    Keep a relationship type if there exists at least one node n:L
    with more than one outgoing relationship of that type.
    (Condition (ii) about target in-degree is dropped.)
    """
    lbl = _backtick(label)
    q = f"""
    MATCH (n:{lbl})-[r]-()
    WITH n, type(r) AS reltype
    // For each (n, reltype), count how many outgoing edges of that type n has
    WITH reltype, n, count(*) AS deg
    WHERE deg > 1
    RETURN DISTINCT reltype
    """
    res = session.run(q)
    return [r["reltype"] for r in res]



def find_only_many_to_many_reltypes(session: Session, label: str) -> List[str]:
    """
    Find relationship types r such that there exists:
      - at least one node n:L with out-degree via r > 1, and
      - at least one neighbor node m (any label) with in-degree via r > 1.
    This detects many-to-many at the instance level.
    """
    lbl = _backtick(label)
    q = f"""
    // Collect candidate reltypes touching L at all
    MATCH (n:{lbl})-[r]->()
    WITH DISTINCT type(r) AS reltype

    // There exists an L-node with degree>1 via reltype?
    WHERE EXISTS {{
      MATCH (n:{lbl})-[r1]->()
      WHERE type(r1) = reltype
      WITH n, count(r1) AS c
      WHERE c > 1
      RETURN 1
    }}

    // And there exists some target node with indegree>1 via reltype?
    AND EXISTS {{
      MATCH ()-[r2]->(m)
      WHERE type(r2) = reltype
      WITH m, count(r2) AS c
      WHERE c > 1
      RETURN 1
    }}

    RETURN reltype
    """
    res = session.run(q)
    return [r["reltype"] for r in res]


def _aggregate_neighbors_for_reltype(
    session: Session,
    label: str,
    reltype: str,
    agg: AggFn = "sum",
    include_rels: bool = True,
    suffixes: Optional[Iterable[str]] = None,
    to_keep=[]
) -> Dict[int, Dict[str, float]]:
    """
    For a given label and relationship type, aggregate numeric properties from:
      1) neighbor nodes' properties
      2) relationship properties (if include_rels=True)
    Returns: { id(n): { column_name: value, ... }, ... }
    """
    lbl = _backtick(label)
    props=to_keep

    # Map Python agg -> Cypher function
    agg_func = {"sum": "sum", "avg": "avg", "min": "min", "max": "max", "count": "count"}[agg]
    #suffixRegex = '(' + '|'.join(map(re.escape, suffixes)) + r')$'
    suffixRegex = '.*(?:' + '|'.join(map(re.escape, suffixes)) + ')$'
    if to_keep==[]:
        # --- Neighbor node properties ---
        q_neighbors = f"""
        MATCH (n:{lbl})-[r]-(m)
        WHERE type(r) = $reltype
        WITH n, m
        UNWIND keys(m) AS k
        WITH n, k, m[k] AS v
        // Keep numeric values only (INTEGER, FLOAT, NUMBER)
        WHERE v IS NOT NULL AND apoc.meta.cypher.type(v) IN ['INTEGER','FLOAT','NUMBER']
        AND NOT k =~ '{suffixRegex}' 
        RETURN id(n) AS nid, k AS prop, {agg_func}(toFloat(v)) AS val
        """
        q_neighbors = f"""
                        MATCH (n:{lbl})-[r]-(m)
                        WHERE type(r) = $reltype
                        WITH n, m
                        UNWIND keys(m) AS k
                        WITH n, k, m[k] AS v
                        // Keep numeric values only (INTEGER, FLOAT, NUMBER)
                        WHERE v IS NOT NULL AND apoc.meta.cypher.type(v) IN ['INTEGER','FLOAT','NUMBER']
                        RETURN id(n) AS nid, k AS prop, {agg_func}(toFloat(v)) AS val
                        """
    else:
        # --- Neighbor node properties ---
        q_neighbors = f"""
                        MATCH (n:{lbl})-[r]-(m)
                        WHERE type(r) = $reltype
                        UNWIND {props} AS k
                        WITH n, k, m[k] AS v
                        // Keep numeric values only (INTEGER, FLOAT, NUMBER)
                        WHERE v IS NOT NULL AND apoc.meta.cypher.type(v) IN ['INTEGER','FLOAT','NUMBER']
                        RETURN id(n) AS nid, k AS prop, {agg_func}(toFloat(v)) AS val
                        """
    neighbor_rows = session.run(q_neighbors, {"reltype": reltype}).data()

    result: Dict[int, Dict[str, float]] = {}
    for row in neighbor_rows:
        nid = row["nid"]
        col = f"{reltype}__neighbor__{agg}_{row['prop']}"
        result.setdefault(nid, {})[col] = row["val"]

    # --- Relationship properties ---
    if include_rels:
        if to_keep== []:
            q_rels = f"""
            MATCH (n:{lbl})-[r]-()
            WHERE type(r) = $reltype
            UNWIND keys(r) AS k
            WITH n, k, r[k] AS v
            WHERE v IS NOT NULL AND apoc.meta.cypher.type(v) IN ['INTEGER','FLOAT','NUMBER']
            AND NOT k =~ '{suffixRegex}'
            RETURN id(n) AS nid, k AS prop, {agg_func}(toFloat(v)) AS val
            """
            q_rels = f"""
                                    MATCH (n:{lbl})-[r]-()
                                    WHERE type(r) = $reltype
                                    UNWIND keys(r) AS k
                                    WITH n, k, r[k] AS v
                                    WHERE v IS NOT NULL AND apoc.meta.cypher.type(v) IN ['INTEGER','FLOAT','NUMBER']
                                    RETURN id(n) AS nid, k AS prop, {agg_func}(toFloat(v)) AS val
                                    """
        else:
            q_rels = f"""
                                    MATCH (n:{lbl})-[r]-()
                                    WHERE type(r) = $reltype
                                    UNWIND {props} AS k
                                    WITH n, k, r[k] AS v
                                    WHERE v IS NOT NULL AND apoc.meta.cypher.type(v) IN ['INTEGER','FLOAT','NUMBER']
                                    RETURN id(n) AS nid, k AS prop, {agg_func}(toFloat(v)) AS val
                                    """
        rel_rows = session.run(q_rels, {"reltype": reltype}).data()
        for row in rel_rows:
            nid = row["nid"]
            col = f"{reltype}__rel__{agg}_{row['prop']}"
            result.setdefault(nid, {})[col] = row["val"]

    return result


def aggregate_m2m_properties_for_label(
    driver: Driver,
    label: str,
    agg: AggFn = "sum",
    include_relationship_properties: bool = True,
    only_reltypes: Optional[Iterable[str]] = None,
    suffixes: Optional[Iterable[str]] = None,
    to_keep = None
) -> pd.DataFrame:
    """
    High-level function:
      1) Finds many-to-many relationship types touching nodes with label `label`
         (unless `only_reltypes` is provided).
      2) For each such reltype, aggregates numeric neighbor and relationship properties.
      3) Returns a pandas DataFrame with one row per node (id(n)) and aggregated columns.

    Args:
        driver: neo4j.Driver (you said you already have connection code;
                pass the driver you already create)
        label: the node label L
        agg: 'sum' (default), 'avg', 'min', 'max', or 'count'
        include_relationship_properties: include numeric properties from the relationships themselves
        only_reltypes: if provided, skip detection and restrict to these reltypes

    Returns:
        pandas.DataFrame with columns: ['node_id', <aggregated columns>...]
    """
    with driver.session() as session:
        reltypes = list(only_reltypes) if only_reltypes else find_many_to_many_reltypes(session, label)
        #print(reltypes)

        # If no M2M reltypes, still return DF with node ids and no extra cols
        if not reltypes:
            ids = session.run(f"MATCH (n:{_backtick(label)}) RETURN id(n) AS nid").data()
            df = pd.DataFrame({"node_id": [r["nid"] for r in ids]})
            return df

        per_node_maps: Dict[int, Dict[str, float]] = {}

        for rt in reltypes:
            partial = _aggregate_neighbors_for_reltype(
                session, label, rt, agg=agg, include_rels=include_relationship_properties, suffixes=suffixes, to_keep=to_keep
            )
            # Merge maps
            for nid, cols in partial.items():
                per_node_maps.setdefault(nid, {}).update(cols)

        # Ensure we include all nodes with label, even if they had no neighbors
        ids = session.run(f"MATCH (n:{_backtick(label)}) RETURN id(n) AS nid").data()
        all_ids = [r["nid"] for r in ids]
        for nid in all_ids:
            per_node_maps.setdefault(nid, {})

        # Build DataFrame
        rows = []
        for nid, cols in per_node_maps.items():
            row = {"node_id": nid}
            row.update(cols)
            rows.append(row)

        df = pd.DataFrame(rows).sort_values("node_id").reset_index(drop=True)

        # For sum/avg/min/max it's often practical to fill NaN with 0 (sum) or leave NaN.
        # We'll leave NaN so you can distinguish “no data” vs “zero”.
        return df


# -------- Example usage (you already have connection code) --------
if __name__ == "__main__":
    """
    Example:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j","password"))

        # Aggregate neighbor and relationship numeric properties for label 'Airport'
        df = aggregate_m2m_properties_for_label(driver, "Airport", agg="sum", include_relationship_properties=True)

        print(df.head())
    """
    pass
