import json
from neo4j import GraphDatabase

# --- Connection Details for your Local Neo4j DB ---
URI = "neo4j://127.0.0.1:7687"
#AUTH = ("neo4j", "kg_bim123")  # duplex
#AUTH = ("neo4j", "kg_bim_cranehall")  # cranehall
AUTH = ("neo4j", "riedelbau_model")  # riedelbau_model

# --- Helper to flatten properties ---
def flatten_properties(props, prefix=""):
    flat = {}
    for k, v in props.items():
        if isinstance(v, dict):
            # Recurse into nested dicts
            nested = flatten_properties(v, prefix=f"{prefix}{k}_")
            flat.update(nested)
        elif isinstance(v, list):
            # If it's a list of primitives, keep it
            if all(isinstance(i, (int, float, str, bool)) for i in v):
                flat[f"{prefix}{k}"] = v
            else:
                # Convert list of dicts/objects into JSON string
                flat[f"{prefix}{k}"] = json.dumps(v)
        else:
            flat[f"{prefix}{k}"] = v
    return flat

def import_data(driver, graph_data):
    # Create all nodes
    for node in graph_data["nodes"]:
        # Copy properties except 'label'
        properties = {k: v for k, v in node.items() if k != 'label'}
        # Flatten all nested dicts/lists
        properties = flatten_properties(properties)

        cypher_query = f"""
        MERGE (n:{node['label']} {{ifcId: $ifcId}})
        SET n += $properties
        """
        driver.execute_query(cypher_query, ifcId=node['ifcId'], properties=properties)
    
    # Create all relationships
    for rel in graph_data["relationships"]:
        cypher_query = f"""
        MATCH (a {{ifcId: $source}})
        MATCH (b {{ifcId: $target}})
        MERGE (a)-[r:{rel['label']}]->(b)
        """
        driver.execute_query(cypher_query, source=rel['source'], target=rel['target'])

if __name__ == "__main__":
    with open('bim_riedel.json', 'r') as f:
        bim_data = json.load(f)

    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()
        print("Database connection successful. Starting import...")
        import_data(driver, bim_data)
        print("Import complete.")
