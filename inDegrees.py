from typing import Optional
import pandas as pd

def in_degree_by_relationship_type(driver, label: str) -> pd.DataFrame:
    """
    Compute, for every node with the given label, the in-degree grouped by relationship type.
    Returns a pandas DataFrame with:
      - first column: nodeId (Neo4j internal id)
      - subsequent columns: one per relationship type, values are counts (integers)

    Parameters
    ----------
    driver : neo4j.Driver
        An existing Neo4j driver (e.g., from neo4j import GraphDatabase; GraphDatabase.driver(...))
    label : str
        Node label to target (e.g., "City")

    Notes
    -----
    - Only incoming relationships are counted: ()-[r]->(n:Label).
    - Nodes with no incoming relationships are included with zeros across all relationship columns.
    - Relationship types become column names.
    """
    # Safely backtick-escape the label (handles spaces/special chars)
    label_bt = f"`{label}`"

    q_nodes = f"""
    MATCH (n:{label_bt})
    RETURN id(n) AS nodeId
    """

    q_counts = f"""
    MATCH ()-[r]->(n:{label_bt})
    RETURN id(n) AS nodeId, type(r) AS relType, count(r) AS cnt
    """

    with driver.session() as session:
        # All target node ids
        node_rows = session.run(q_nodes).data()
        nodes_df = pd.DataFrame(node_rows)
        if nodes_df.empty:
            # No nodes of that label: return empty frame with nodeId only
            return pd.DataFrame(columns=["nodeId"]).astype({"nodeId": "Int64"})

        # In-degree counts grouped by relationship type
        count_rows = session.run(q_counts).data()
        counts_df = pd.DataFrame(count_rows, columns=["nodeId", "relType", "cnt"])

    if counts_df.empty:
        # No incoming relationships at all; return zeros-only columns (just nodeId)
        df = nodes_df.copy()
        df = df.astype({"nodeId": "Int64"})
        return df

    # Pivot so each relationship type is its own column
    pivot = counts_df.pivot_table(
        index="nodeId", columns="relType", values="cnt", aggfunc="sum", fill_value=0
    ).reset_index()

    # Ensure *all* nodes are present (including those without incoming edges)
    df = nodes_df.merge(pivot, on="nodeId", how="left").fillna(0)

    # Make integer dtype for counts when possible
    for col in df.columns:
        if col != "nodeId":
            df[col] = df[col].astype(int)

    # Sort columns: nodeId first, then relationship types (alphabetical)
    cols = ["nodeId"] + sorted([c for c in df.columns if c != "nodeId"])
    df = df[cols].astype({"nodeId": "Int64"})

    return df
