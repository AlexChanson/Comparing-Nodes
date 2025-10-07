from typing import Dict, List, Set, Tuple, Optional
from collections import deque
from neo4j import GraphDatabase

def schema_hops_from_label(
    driver,
    start_label: str,
    *,
    include_relationship_types: bool = True,
    directed: bool = False,
    fallback_from_data: bool = True,
) -> Dict[str, Dict[str, Optional[int]]]:
    """
    Compute schema hop counts from a given node label to all other schema objects.

    Parameters
    ----------
    driver : neo4j.Driver
        Active Neo4j driver (Neo4j 5.x).
    start_label : str
        The node label to start from (e.g., "Airport").
    include_relationship_types : bool, default True
        If True, also compute distance to every relationship type as:
        1 + min(distance to either of its endpoint labels). If an endpoint
        is unreachable, the rel-type distance is None.
    directed : bool, default False
        If True, build a directed schema graph using (startLabels -> endLabels).
        If False (default), treat it as undirected (usual for schema reachability).
    fallback_from_data : bool, default True
        If APOC's schema returns no relationships, optionally derive the schema
        directly from data with a lightweight DISTINCT scan. Disable if your
        graph is huge and you prefer to fail fast.

    Returns
    -------
    Dict[str, Dict[str, Optional[int]]]
        {
          "labels": { "<Label>": int|None, ... },                     # hops from start_label
          "relationships": { "<REL_TYPE>": int|None, ... } (optional) # hops via rel-type as a node
        }
        Distances are hop counts; unreachable -> None. `labels[start_label]` is 0.
    """
    def _normalize(s: Optional[str]) -> Optional[str]:
        if s is None: return None
        # strip accidental backticks/colons (can appear in some meta outputs)
        return s.replace('`', '').replace(':', '')

    # --- 1) Pull schema using APOC ---
    with driver.session() as sess:
        rec = sess.run("CALL apoc.meta.schema() YIELD value RETURN value").single()
        value = rec["value"] if rec else {}
        rels_map = value.get("relationships", {}) or {}

        # If APOC reports nothing and fallback is allowed, derive schema from data.
        if not rels_map and fallback_from_data:
            # Build edges and endpoints from actual data (distinct label pairs and rel types)
            data_rows = sess.run("""
                MATCH (a)-[r]->(b)
                WITH labels(a) AS la, type(r) AS t, labels(b) AS lb
                UNWIND la AS A
                UNWIND lb AS B
                RETURN A AS a, B AS b, t AS t
            """)
            edges = [( _normalize(r["a"]), _normalize(r["b"]), _normalize(r["t"]) ) for r in data_rows]
        else:
            # Expand APOC schema into (a,b,t) triples
            edges: List[Tuple[str, str, str]] = []
            for t, spec in rels_map.items():
                t_norm = _normalize(t)
                for a in spec.get("startLabels", []) or []:
                    a_norm = _normalize(a)
                    for b in spec.get("endLabels", []) or []:
                        b_norm = _normalize(b)
                        if a_norm and b_norm and t_norm:
                            edges.append((a_norm, b_norm, t_norm))

    # --- 2) Build label set + adjacency ---
    label_set: Set[str] = set()
    for a, b, _ in edges:
        if a: label_set.add(a)
        if b: label_set.add(b)

    # Ensure the start label is represented even if isolated
    start_label = _normalize(start_label)
    if start_label:
        label_set.add(start_label)

    # adjacency: label -> set(neighbor labels)
    adj: Dict[str, Set[str]] = {lbl: set() for lbl in label_set}
    for a, b, _ in edges:
        if a and b and a in adj and b in adj:
            adj[a].add(b)
            if not directed:
                adj[b].add(a)

    # --- 3) BFS from start_label over labels ---
    def bfs(start: str) -> Dict[str, Optional[int]]:
        dist: Dict[str, Optional[int]] = {lbl: None for lbl in adj.keys()}
        if start not in adj:
            # Start label not present in schema graph; only itself at distance 0
            dist[start] = 0
            return dist
        q = deque([start])
        dist[start] = 0
        while q:
            u = q.popleft()
            for v in adj.get(u, ()):
                if dist[v] is None:
                    dist[v] = dist[u] + 1 if dist[u] is not None else 1
                    q.append(v)
        return dist

    label_dist = bfs(start_label)

    # --- 4) Relationship-type distances (optional) ---
    reltype_dist: Dict[str, Optional[int]] = {}
    if include_relationship_types:
        # endpoints per relationship type
        endpoints: Dict[str, Set[str]] = {}
        for a, b, t in edges:
            if not t:
                continue
            s = endpoints.setdefault(t, set())
            if a: s.add(a)
            if b: s.add(b)

        for t, eps in endpoints.items():
            # Distance to rel-type is 1 + min distance to any endpoint label
            best = None
            for lbl in eps:
                d = label_dist.get(lbl)
                if d is not None:
                    best = d if best is None else min(best, d)
            reltype_dist[t] = (None if best is None else best + 1)

    result = {"labels": label_dist}
    if include_relationship_types:
        result["relationships"] = reltype_dist
    return result


import pandas as pd
from pandas.api.types import is_numeric_dtype
from typing import Dict, Optional, Iterable

def weight_df_by_schema_hops(
    df: pd.DataFrame,
    hops_result: Dict[str, Dict[str, Optional[int]]],
    *,
    id_first_col: bool = True,
    columns_to_skip: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Scale dataframe columns by 1/(h+1) where h is the schema hop count for
    the graph element (node label or relationship type) that prefixes the column name.

    Assumptions
    -----------
    - df's first column is an identifier (left unchanged) if id_first_col=True.
    - Each data column name starts with a graph element name, then an underscore,
      then anything else (e.g., 'Country_physicians', 'ROUTE_TO_avg_delay', etc.).
    - hops_result is the dict returned by `schema_hops_from_label(...)`, i.e.:
        {
          "labels": { "Airport": 0, "City": 1, "Country": 2, ... },
          "relationships": { "IN_COUNTRY": 3, ... }  # optional
        }

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    hops_result : dict
        Output from schema_hops_from_label(...).
    id_first_col : bool, default True
        If True, the first column is treated as an identifier and never scaled.
    columns_to_skip : optional iterable of str
        Column names to leave unchanged.

    Returns
    -------
    pd.DataFrame
        New dataframe with identical structure (same columns, same row count),
        but eligible columns scaled by 1/(h+1). Non-numeric columns are left as-is.
    """
    columns_to_skip = set(columns_to_skip or [])

    # Merge label + relationship hop maps
    hop_map: Dict[str, Optional[int]] = {}
    hop_map.update({str(k): v for k, v in (hops_result.get("labels") or {}).items()})
    hop_map.update({str(k): v for k, v in (hops_result.get("relationships") or {}).items()})

    # Prefer the longest element name first (to handle names with underscores, e.g., ROUTE_TO)
    element_names = sorted(hop_map.keys(), key=len, reverse=True)

    out = df.copy()
    cols = list(out.columns)
    start_idx = 1 if (id_first_col and len(cols) > 0) else 0

    for col in cols[start_idx:]:
        if col in columns_to_skip:
            continue

        # Find the longest graph element name K such that column starts with "K_"
        match_name = None
        for name in element_names:
            if col.startswith(name + "_"):
                match_name = name
                break
        if match_name is None:
            continue  # no matching graph element prefix

        h = hop_map.get(match_name)
        coeff = 1.0 if h is None else 1.0 / (h + 1.0)

        # Scale only numeric columns; try gentle coercion if it's mixed/strings
        if is_numeric_dtype(out[col]):
            out[col] = out[col] * coeff
        else:
            coerced = pd.to_numeric(out[col], errors="coerce")
            if coerced.notna().any():
                out[col] = coerced * coeff
            # else: leave the non-numeric column unchanged

    return out



if __name__ == "__main__":

    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j","airports"))

    dist = schema_hops_from_label(driver, "Airport", include_relationship_types=True, directed=False)
    print(dist)
    print("Label hops from Airport:")
    for k, v in sorted(dist["labels"].items()):
        print(f"  {k:20s} -> {v}")

    print("\nRel-type hops from Airport:")
    for k, v in sorted(dist["relationships"].items()):
        print(f"  {k:20s} -> {v}")
