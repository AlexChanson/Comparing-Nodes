import argparse
import csv
import json
from collect_indicators import Neo4jConnector, DatabaseConfig
from orchestrate_neo4j import start_dbms, stop_dbms, wait_for_bolt

def generate_m2m_template(config_path: str, output_csv: str):
    # 1. Load configuration
    with open(config_path) as f:
        # Ignore any extra arguments in the JSON not recognized by DatabaseConfig
        config_dict = json.load(f)
        db_config = DatabaseConfig(**{k: v for k, v in config_dict.items() if k in DatabaseConfig.__match_args__ or k in DatabaseConfig.__dataclass_fields__})

    print(f"Starting database: {db_config.name}...")
    db_spec = db_config.get_db_spec()
    start_dbms(db_spec)
    wait_for_bolt(db_spec.bolt_uri, db_spec.user, db_spec.password, db_spec.start_timeout)

    print("Connection established. Analyzing cardinalities to target the Many-to-Many context...")
    
    with Neo4jConnector(db_config.uri, db_config.username, db_config.password) as db:
        # Step A: Identify exact Many-to-Many relationships using the existing function
        manyToOne, manyToMany = db.detect_relationship_cardinalities()

        if not manyToMany:
            print("No Many-to-Many relationships detected. No attributes will require aggregation.")
            stop_dbms(db_spec)
            return

        print(f"Detected M2M relationships: {manyToMany}")
        start_labels = list(db_config.labels)

        # Step B: Find labels of neighbor nodes connected via these M2M relationships
        query_targets = """
        MATCH (n)-[r]-(m)
        WHERE any(lbl IN labels(n) WHERE lbl IN $start_labels)
          AND type(r) IN $m2m_rels
        UNWIND labels(m) AS target_label
        RETURN DISTINCT target_label
        """
        target_results = db.execute_query(query_targets, {"start_labels": start_labels, "m2m_rels": manyToMany})
        target_labels = [rec["target_label"] for rec in target_results]

        print(f"Neighbor node labels identified in the M2M context: {target_labels}")

        if not target_labels:
            print("No neighbor nodes found at the end of these relationships. Stopping.")
            stop_dbms(db_spec)
            return

        # Step C: Retrieve only numeric attributes from these targets via APOC
        query_props = """
        CALL apoc.meta.nodeTypeProperties() YIELD nodeType, propertyName, propertyTypes
        // APOC often returns labels in the format :`Label`, so we clean them up
        WITH replace(replace(nodeType, '`', ''), ':', '') AS cleanNodeLabel, propertyName, propertyTypes
        WHERE cleanNodeLabel IN $target_labels
          AND any(t IN propertyTypes WHERE t IN ['INTEGER', 'FLOAT', 'Long', 'Double', 'Number'])
        RETURN DISTINCT propertyName AS attribute
        
        UNION
        
        CALL apoc.meta.relTypeProperties() YIELD relType, propertyName, propertyTypes
        WITH replace(replace(relType, '`', ''), ':', '') AS cleanRelLabel, propertyName, propertyTypes
        WHERE cleanRelLabel IN $m2m_rels
          AND any(t IN propertyTypes WHERE t IN ['INTEGER', 'FLOAT', 'Long', 'Double', 'Number'])
        RETURN DISTINCT propertyName AS attribute
        """
        
        results = db.execute_query(query_props, {
            "target_labels": target_labels, 
            "m2m_rels": manyToMany
        })
        
        # Clean up and sort alphabetically
        attributes = sorted(list(set([record["attribute"] for record in results])))

    print(f"Found {len(attributes)} numeric attributes requiring potential aggregation.")
    
    # 2. Create the CSV template file
    print(f"Creating template file {output_csv}...")
    with open(output_csv, mode='w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['attribute', 'aggregation_function']) 
        
        # Write rows with empty aggregation function column
        for attr in attributes:
            writer.writerow([attr, ''])
            
    print("Stopping database...")
    stop_dbms(db_spec)
            
    print(f"Done! You can now configure your functions (min, max, avg, sum, count) in '{output_csv}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates a CSV template for M2M aggregation functions.")
    parser.add_argument('config', help="Path to the database JSON configuration file")
    parser.add_argument('-o', '--output', default='aggregation_template.csv', help="Output CSV file name (default: aggregation_template.csv)")
    
    args = parser.parse_args()
    
    generate_m2m_template(args.config, args.output)