import argparse
from collect_indicators import Neo4jConnector, DatabaseConfig
import json
from orchestrate_neo4j import start_dbms, stop_current_dbms
from typing import List, Tuple, Any

# type of the returned context elements
ContextTuple = Tuple[str, str, str, Any]   # (type, label, property, value)


import random

def get_random_node_identifier(label: str, db):
    """
    Returns the 'identifier' property of a random node with the given label.

    Parameters
    ----------
    label : str
        Neo4j label of the node (e.g., "CITY", "AIRPORT").
    execute_query : callable
        Function to run Cypher queries.

    Returns
    -------
    The identifier of a randomly selected node, or None if no node exists.
    """

    cypher = f"""
    MATCH (n:`{label}`)
    RETURN n.identifier AS id
    """

    rows = db.execute_query(cypher, {})

    if not rows:
        return None

    # Extract identifiers
    ids = [row["id"] for row in rows if row["id"] is not None]

    if not ids:
        return None

    return random.choice(ids)


from typing import List, Tuple, Any, Dict

ContextTuple = Tuple[str, List[str], str, str, Any]
#                (type,   source,   label, property, value)


def get_node_context(node_id: int, allowed_rels: List[str], db) -> List[ContextTuple]:
    """
    Retrieve the context of a node in a Neo4j property graph.

    Parameters
    ----------
    node_id : int
        Neo4j internal id of the start node (id(n)).
    allowed_rels : list of str
        Relationship types allowed when expanding directed outgoing paths
        (e.g., ["IN", "OF"]).
    db : any
        An object exposing a method `execute_query(cypher, parameters)`.

    Returns
    -------
    List[ContextTuple]
        A list of (type, source, label, property, value) tuples where:
          - type   ∈ {"node", "edge"}
          - source is a list indicating the provenance from the initial
            node, e.g. ['IN'] or ['IN','City','OF']
          - label  is the node label(s) (as "A:B") or the relationship type
          - property is the property name
          - value   is the property value
    """

    # ---------------------------------------------
    # 1) Base context: start node, direct neighbours, incident relationships
    # ---------------------------------------------
    base_cypher = """
    MATCH (n)
    WHERE id(n) = $id

    OPTIONAL MATCH (n)-[r0]-(nb)
    WITH n,
         collect(DISTINCT {
             kind:   'node',
             id:     id(nb),
             labels: labels(nb),
             props:  properties(nb),
             source: [type(r0)]
         }) AS directNodes,
         collect(DISTINCT {
             kind:   'rel',
             id:     id(r0),
             label:  type(r0),
             props:  properties(r0),
             source: [type(r0)]
         }) AS incidentRels

    RETURN
      { kind:   'node',
        id:     id(n),
        labels: labels(n),
        props:  properties(n),
        source: []
      } AS startNode,
      directNodes,
      incidentRels
    """

    base_rows = db.execute_query(base_cypher, {"id": node_id})
    if not base_rows:
        return []

    base_rec = base_rows[0]
    start_node = base_rec["startNode"]                 # map
    direct_nodes = base_rec.get("directNodes", [])     # list of maps
    incident_rels = base_rec.get("incidentRels", [])   # list of maps

    # ---------------------------------------------
    # Aggregator: keep shortest provenance per (element, property)
    # ---------------------------------------------
    best: Dict[Tuple[str, int, str], Tuple[List[str], str, Any]] = {}

    def consider(
        kind: str,
        element_id: int,
        source: List[str],
        label: str,
        props: Dict[str, Any],
    ):
        """Keep the shortest provenance list per (kind, id, property)."""
        if props is None:
            return
        for prop, value in props.items():
            key = (kind, element_id, prop)
            if key not in best or len(source) < len(best[key][0]):
                best[key] = (source, label, value)

    def label_str_from_labels(labels) -> str:
        if isinstance(labels, (list, tuple)):
            return ":".join(labels)
        return str(labels) if labels is not None else ""

    # Start node
    consider(
        kind=start_node["kind"],
        element_id=start_node["id"],
        source=start_node.get("source", []),
        label=label_str_from_labels(start_node.get("labels")),
        props=start_node.get("props"),
    )

    # Direct neighbour nodes
    for nd in direct_nodes:
        consider(
            kind=nd.get("kind", "node"),
            element_id=nd.get("id"),
            source=nd.get("source", []),
            label=label_str_from_labels(nd.get("labels")),
            props=nd.get("props"),
        )

    # Incident relationships
    for rd in incident_rels:
        consider(
            kind=rd.get("kind", "rel"),
            element_id=rd.get("id"),
            source=rd.get("source", []),
            label=str(rd.get("label", "")),
            props=rd.get("props"),
        )

    # ---------------------------------------------
    # 2) Path expansion with APOC for allowed_rels
    # ---------------------------------------------
    if allowed_rels:
        rel_filter = "|".join(r + ">" for r in allowed_rels)

        path_cypher = """
        MATCH (n)
        WHERE id(n) = $id

        CALL apoc.path.expandConfig(n, {
            relationshipFilter: $allowedFilter,
            minLevel: 1,
            uniqueness: 'NODE_GLOBAL'
        }) YIELD path

        WITH nodes(path) AS nds, relationships(path) AS rels
        RETURN
          [x IN nds |
             { id:     id(x),
               labels: labels(x),
               props:  properties(x)
             }
          ] AS pathNodes,
          [r IN rels |
             { id:    id(r),
               type:  type(r),
               props: properties(r)
             }
          ] AS pathRels
        """

        path_rows = db.execute_query(path_cypher, {
            "id": node_id,
            "allowedFilter": rel_filter,
        })

        # Helper: provenance functions (in Python, simpler & safer)
        def node_provenance(rel_types: List[str],
                            node_labels: List[str],
                            j: int) -> List[str]:
            """
            Provenance for node at index j (j >= 1) in the path:
            rels = [r0, r1, ...], labels = [L0, L1, L2, ...]
            Node 1 (City) via Airport-[:IN]->City:
              ['IN']
            Node 2 (Country) via Airport-[:IN]->City-[:OF]->Country:
              ['IN', 'City', 'OF']
            Formula:
              [r0] + flatten( [ [Lk, rk] for k in 1..j-1 ] )
            """
            if not rel_types:
                return []
            seq: List[str] = [rel_types[0]]
            for k in range(1, j):
                seq.append(node_labels[k])
                seq.append(rel_types[k])
            return seq

        def rel_provenance(rel_types: List[str],
                           node_labels: List[str],
                           i: int) -> List[str]:
            """
            Provenance for relationship at index i (0-based) in the path:
            Relationship 0 (IN) on Airport-[:IN]->City:
              ['IN']
            Relationship 1 (OF) on City-[:OF]->Country:
              ['IN', 'City', 'OF']
            Formula:
              [r0] + flatten( [ [Lk, rk] for k in 1..i ] )
            """
            if not rel_types:
                return []
            seq: List[str] = [rel_types[0]]
            for k in range(1, i + 1):
                seq.append(node_labels[k])
                seq.append(rel_types[k])
            return seq

        # Process each path
        for row in path_rows:
            path_nodes = row["pathNodes"]  # list of maps
            path_rels = row["pathRels"]    # list of maps

            if not path_rels:
                continue  # no edges => nothing beyond the start node

            rel_types = [r["type"] for r in path_rels]
            node_labels = [
                label_str_from_labels(n.get("labels")) for n in path_nodes
            ]

            # Nodes: indices 1..len(path_nodes)-1
            for j in range(1, len(path_nodes)):
                nm = path_nodes[j]
                prov = node_provenance(rel_types, node_labels, j)
                consider(
                    kind="node",
                    element_id=nm["id"],
                    source=prov,
                    label=node_labels[j],
                    props=nm.get("props"),
                )

            # Relationships: indices 0..len(path_rels)-1
            for i, rm in enumerate(path_rels):
                prov = rel_provenance(rel_types, node_labels, i)
                consider(
                    kind="rel",
                    element_id=rm["id"],
                    source=prov,
                    label=rm["type"],
                    props=rm.get("props"),
                )

    # ---------------------------------------------
    # 3) Build final list of tuples
    # ---------------------------------------------
    context: List[ContextTuple] = []
    for (kind, _elem_id, prop), (source, label, value) in best.items():
        type_str = "node" if kind == "node" else "edge"
        context.append((type_str, source, label, prop, value))

    return context



def main(database_config: DatabaseConfig, size) -> None:
    print("database: ", database_config.name)
    #stop_current_dbms()
    #db_spec = database_config.get_db_spec()
    #start_dbms(db_spec)


    with Neo4jConnector(database_config.uri, database_config.username, database_config.name) as db:
        # finds relationship cardinalities
        print("Finding cardinalities")
        manyToOne, manyToMany = db.detect_relationship_cardinalities()
        print(manyToOne)
        print(manyToMany)

        id=get_random_node_identifier("AIRPORT", db)
        #orly = 144786
        #Tours = 144776
        id=144776
        context=get_node_context(id,manyToOne,db)
        print("context: ", context)

    #stop_current_dbms()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('config')
    parser.add_argument('-s', '--size', default=1, type=int) # context size in nb of hops


    args = parser.parse_args()

    with open(args.config) as f:
        database_config = json.load(f, object_hook=lambda x: DatabaseConfig(**x))

    main(
        database_config=database_config,
        size=args.size
    )
