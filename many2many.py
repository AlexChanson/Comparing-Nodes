import re
from collections.abc import Iterable
from typing import Dict, List, Literal, Optional, Union

import pandas as pd
from neo4j import Driver, Session

AggFn = Literal["sum", "avg", "min", "max", "count"]


def _backtick(label: str) -> str:
    return f"`{label.replace('`', '``')}`"


def find_many_to_many_reltypes(session: Session, label: str) -> List[str]:
    lbl = _backtick(label)
    q = f"""
    MATCH (n:{lbl})-[r]-()
    WITH n, type(r) AS reltype
    WITH reltype, n, count(*) AS deg
    WHERE deg > 1
    RETURN DISTINCT reltype
    """
    res = session.run(q)
    return [r["reltype"] for r in res]


def find_only_many_to_many_reltypes(session: Session, label: str) -> List[str]:
    lbl = _backtick(label)
    q = f"""
    MATCH (n:{lbl})-[r]->()
    WITH DISTINCT type(r) AS reltype

    WHERE EXISTS {{
      MATCH (n:{lbl})-[r1]->()
      WHERE type(r1) = reltype
      WITH n, count(r1) AS c
      WHERE c > 1
      RETURN 1
    }}

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
    agg: Union[AggFn, Dict[str, str]] = "sum",
    include_rels: bool = True,
    suffixes: Optional[Iterable[str]] = None,
    to_keep=[],
) -> Dict[int, Dict[str, float]]:
    
    lbl = _backtick(label)
    props = to_keep

    cypher_aggregations = """
        sum(toFloat(v)) AS val_sum, 
        avg(toFloat(v)) AS val_avg, 
        min(toFloat(v)) AS val_min, 
        max(toFloat(v)) AS val_max, 
        count(toFloat(v)) AS val_count
    """

    if to_keep == []:
        q_neighbors = f"""
            MATCH (n:{lbl})-[r]-(m)
            WHERE type(r) = $reltype
            WITH n, m
            UNWIND keys(m) AS k
            WITH n, k, m[k] AS v
            WHERE v IS NOT NULL AND apoc.meta.cypher.type(v) IN ['INTEGER', 'FLOAT', 'Number', 'Long', 'Double']
            RETURN id(n) AS nid, k AS prop, {cypher_aggregations}
        """
    else:
        q_neighbors = f"""
            MATCH (n:{lbl})-[r]-(m)
            WHERE type(r) = $reltype
            UNWIND {props} AS k
            WITH n, k, m[k] AS v
            WHERE v IS NOT NULL AND apoc.meta.cypher.type(v) IN ['INTEGER', 'FLOAT', 'Number', 'Long', 'Double']
            RETURN id(n) AS nid, k AS prop, {cypher_aggregations}
        """
        
    neighbor_rows = session.run(q_neighbors, {"reltype": reltype}).data()
    result: Dict[int, Dict[str, float]] = {}
    
    def get_val_for_prop(row, prop_name):
        if isinstance(agg, dict):
            func = agg.get(prop_name, "avg").lower()
        else:
            func = agg.lower()
            
        if func not in ["sum", "avg", "min", "max", "count"]:
            func = "avg"
            
        return row[f"val_{func}"], func

    for row in neighbor_rows:
        nid = row["nid"]
        prop = row["prop"]
        val, func_used = get_val_for_prop(row, prop)
        
        col = f"{reltype}__neighbor__{func_used}_{prop}"
        result.setdefault(nid, {})[col] = val

    if include_rels:
        if to_keep == []:
            q_rels = f"""
                MATCH (n:{lbl})-[r]-()
                WHERE type(r) = $reltype
                UNWIND keys(r) AS k
                WITH n, k, r[k] AS v
                WHERE v IS NOT NULL AND apoc.meta.cypher.type(v) IN ['INTEGER', 'FLOAT', 'Number', 'Long', 'Double']
                RETURN id(n) AS nid, k AS prop, {cypher_aggregations}
            """
        else:
            q_rels = f"""
                MATCH (n:{lbl})-[r]-()
                WHERE type(r) = $reltype
                UNWIND {props} AS k
                WITH n, k, r[k] AS v
                WHERE v IS NOT NULL AND apoc.meta.cypher.type(v) IN ['INTEGER', 'FLOAT', 'Number', 'Long', 'Double']
                RETURN id(n) AS nid, k AS prop, {cypher_aggregations}
            """
            
        rel_rows = session.run(q_rels, {"reltype": reltype}).data()
        for row in rel_rows:
            nid = row["nid"]
            prop = row["prop"]
            val, func_used = get_val_for_prop(row, prop)
            
            col = f"{reltype}__rel__{func_used}_{prop}"
            result.setdefault(nid, {})[col] = val

    return result


def aggregate_m2m_properties_for_label(
    driver: Driver,
    label: str,
    database: str = "neo4j",
    agg: Union[AggFn, Dict[str, str]] = "sum",
    include_relationship_properties: bool = True,
    only_reltypes: Optional[Iterable[str]] = None,
    suffixes: Optional[Iterable[str]] = None,
    to_keep=None,
) -> pd.DataFrame:

    with driver.session(database=database) as session:
        reltypes = list(only_reltypes) if only_reltypes else find_many_to_many_reltypes(session, label)

        if not reltypes:
            ids = session.run(f"MATCH (n:{_backtick(label)}) RETURN id(n) AS nid").data()
            return pd.DataFrame({"node_id": [r["nid"] for r in ids]})

        per_node_maps: Dict[int, Dict[str, float]] = {}

        for rt in reltypes:
            partial = _aggregate_neighbors_for_reltype(
                session,
                label,
                rt,
                agg=agg,
                include_rels=include_relationship_properties,
                suffixes=suffixes,
                to_keep=to_keep,
            )
            for nid, cols in partial.items():
                per_node_maps.setdefault(nid, {}).update(cols)

        ids = session.run(f"MATCH (n:{_backtick(label)}) RETURN id(n) AS nid").data()
        all_ids = [r["nid"] for r in ids]
        for nid in all_ids:
            per_node_maps.setdefault(nid, {})

        rows = []
        for nid, cols in per_node_maps.items():
            row = {"node_id": nid}
            row.update(cols)
            rows.append(row)

        if not rows:
            return pd.DataFrame(columns=["node_id"])

        return pd.DataFrame(rows).sort_values("node_id").reset_index(drop=True)


if __name__ == "__main__":
    pass