# schema_distances_session.py
# Compute schema distances from a start node label to all labels and relationship types
# using db.schema.visualization() only (instance-free). Traversal is undirected by default.

from collections import deque, defaultdict
from typing import Dict, Iterable, Set
import pandas as pd
from neo4j import Session


def _single_label_from_schema_name(name: str) -> str:
    """
    db.schema.visualization() exposes virtual node names like ':Label'
    (under your assumption: one label per schema node).
    """
    if not isinstance(name, str):
        raise ValueError(f"Unexpected schema node name: {name!r}")
    parts = [p for p in name.split(':') if p]  # drop empty head from leading ':'
    if not parts:
        raise ValueError(f"No label parsed from schema node name: {name!r}")
    return parts[0]  # assumption: exactly one label


def fetch_schema_pairs(
    session: Session,
    directed: bool = False,
) -> (Set[str], Set[str], Dict[str, Set[str]]):
    """
    Build the bipartite meta-graph from db.schema.visualization().

    Returns:
      labels   : set of label names
      reltypes : set of relationship type names
      adj      : adjacency map over meta-nodes 'L:<Label>' and 'R:<RELTYPE>'
    """
    cypher = "CALL db.schema.visualization() YIELD nodes, relationships RETURN nodes, relationships"
    rec = session.run(cypher).single()
    if rec is None:
        raise RuntimeError("db.schema.visualization() returned no result.")
    relationships = rec["relationships"]

    labels, reltypes = set(), set()
    adj: Dict[str, Set[str]] = defaultdict(set)

    for r in relationships:
        rt = r["type"]
#        s_label = _single_label_from_schema_name(r["startNode"]["name"])
#        t_label = _single_label_from_schema_name(r["endNode"]["name"])
        s_label = _single_label_from_schema_name(r.start_node["name"])
        t_label = _single_label_from_schema_name(r.end_node["name"])

        labels.update((s_label, t_label))
        reltypes.add(rt)

        ls, rr, lt = f"L:{s_label}", f"R:{rt}", f"L:{t_label}"

        # forward legs
        adj[ls].add(rr)
        adj[rr].add(lt)

        if not directed:
            # make edges undirected in the meta-graph
            adj[rr].add(ls)
            adj[lt].add(rr)

    return labels, reltypes, adj


def bfs_meta_dist(adj: Dict[str, Set[str]], start_key: str) -> Dict[str, int]:
    """Standard BFS on the meta-graph to get meta-depths."""
    dist = {start_key: 0}
    q = deque([start_key])
    while q:
        u = q.popleft()
        for v in adj.get(u, ()):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def schema_distances_from_label(
    session: Session,
    start_label: str,
    directed: bool = False,
) -> pd.DataFrame:
    """
    Compute distances from start_label to all labels and rel-types in the schema.

    Distance convention (real graph hops):
      - labels live at even meta-depths: distance = meta_depth // 2
      - rel-types live at odd  meta-depths: distance = (meta_depth + 1) // 2
    """
    labels, reltypes, adj = fetch_schema_pairs(session, directed=directed)

    if start_label not in labels:
        raise ValueError(f"Start label '{start_label}' not found in schema. Available: {sorted(labels)}")

    start_key = f"L:{start_label}"
    meta_dist = bfs_meta_dist(adj, start_key)

    rows = []
    for lab in sorted(labels):
        key = f"L:{lab}"
        md = meta_dist.get(key)
        dist = (md // 2) if md is not None else None
        rows.append({"type": lab, "kind": "label", "distance": dist})

    for rt in sorted(reltypes):
        key = f"R:{rt}"
        md = meta_dist.get(key)
        dist = ((md + 1) // 2) if md is not None else None
        rows.append({"type": rt, "kind": "relationship", "distance": dist})

    return pd.DataFrame(rows).sort_values(["kind", "type"]).reset_index(drop=True)


def schema_distances_for_targets(
    session: Session,
    start_label: str,
    targets: Iterable[str],
    directed: bool = False,
) -> pd.DataFrame:
    """
    Convenience: compute distances and filter to a target set (mix of labels and rel-types).
    """
    df = schema_distances_from_label(session, start_label, directed=directed)
    return df[df["type"].isin(set(targets))].reset_index(drop=True)


# ------------------ Example usage ------------------
if __name__ == "__main__":
    # Assuming you already created a session:
    # from neo4j import GraphDatabase
    # driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j","password"))
    # with driver.session() as session:
    #     df = schema_distances_from_label(session, start_label="t")
    #     print(df.to_string(index=False))
    pass
