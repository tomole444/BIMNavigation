from typing import List, Dict, Any, Optional
import json
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from neo4j import GraphDatabase
from pathlib import Path

class retrieverUtils:

    def __init__(self, neo4j_uri: str, neo4j_auth: tuple, vectorstore: Chroma):
        self.neo4j_uri = neo4j_uri
        self.neo4j_auth = neo4j_auth
        self.vectorstore = vectorstore
        self.llm = OllamaLLM(model="llama3.2-vision:11b", temperature=0.3)
        object.__setattr__(self, 'driver', GraphDatabase.driver(neo4j_uri, auth=neo4j_auth))


    def get_complete_graph_schema(self) -> str:
        """Get complete schema without limits."""
        schema_parts = []
        
        with self.driver.session() as sess:
            # Get all IFC types with ALL their properties
            query = """
            MATCH (n)
            WHERE n.ifcType IS NOT NULL
            WITH n.ifcType AS type, keys(n) AS props
            UNWIND props AS prop
            WITH type, collect(DISTINCT prop) AS unique_props
            RETURN type, unique_props
            ORDER BY type
            """
            
            schema_parts.append("Complete IFC Schema:")
            schema_parts.append("=" * 50)
            result = sess.run(query)
            for record in result:
                type_name = record["type"]
                props = sorted(record["unique_props"])
                schema_parts.append(f"\n{type_name}:")
                schema_parts.append(f"  Properties: {', '.join(props)}")
            
            # Get all relationship types with frequency
            rel_query = """
            MATCH ()-[r]->()
            RETURN type(r) AS rel_type, count(r) AS count
            ORDER BY count DESC
            """
            schema_parts.append("\n" + "=" * 50)
            schema_parts.append("Relationship Types:")
            for record in sess.run(rel_query):
                schema_parts.append(f"  - {record['rel_type']} ({record['count']} instances)")
        
        return "\n".join(schema_parts)

            

    
    def generate_cypher_from_query(self, query: str, AutoGen = False) -> Optional[str]:

        """Generate Cypher query using LLM."""
        
        schema = self.get_complete_graph_schema()
        
        prompt = f"""  You are an expert in BIM (Building Information Modeling) and Neo4j graph databases.

        I have imported an IFC building model into Neo4j with the following verified schema:

        Nodes represent IFC elements (e.g., IfcDoor, IfcWall, IfcSpace) and have these properties:

        ifcId (string): Unique IFC GlobalId
        ifcType (string): IFC class (e.g., "IfcDoor")
        name (string): Element name (may be "Unnamed")
        materials (list of strings, optional)
        Geometric data (only if geometry was available during import):
        position_x, position_y, position_z (centroid coordinates)
        boundingBox: a list like [minX, minY, minZ, maxX, maxY, maxZ]
        Doors/Windows only: isPassable (boolean), width, height
        Relationships (all directed):

        CONTAINS (e.g., Building → Storey)
        BOUNDED_BY (Space → Wall/Door)
        CONNECTS_TO (Space → Space, with through: ifcId)
        HAS_OPENING (Wall → Door/Window)

        Critical Notes for Query Generation:

        If asked for a natural language answer, reason step-by-step using the schema {schema}.
        If asked for a Cypher query, output only valid, executable Neo4j Cypher that follows the above rules.
        If data might be missing (e.g., no geometry), mention it or use WHERE exists(node.position).

        example queries : 

        # For general querying and exploration

        MATCH p=()-[:CONTAINS]->() RETURN p LIMIT 600;
        MATCH (n:IfcDoor) RETURN n LIMIT 100;

        #To find windows in walls and their positions

        MATCH (container)-[:CONTAINS]->(window)
        WHERE window.ifcType = 'IfcWindow'
        RETURN 
        container.ifcType AS containerType,
        container.name AS containerName,
        window.ifcId AS windowId,
        window.name AS windowName,
        window.position_x,
        window.position_y,
        window.position_z
        
        # To find windows in walls using HAS_OPENING relationship

        MATCH (container)-[:HAS_OPENING]->(window)
        WHERE window.ifcType = 'IfcWindow'
        RETURN 
        container.ifcType AS containerType,
        container.name AS containerName,
        window.ifcId AS windowId,
        window.name AS windowName,
        window.position_x,
        window.position_y,
        window.position_z

        # To find doors in rooms in general

        MATCH (container)-[:CONTAINS]->(door)
        WHERE door.ifcType = 'IfcDoor'
        RETURN 
        container.ifcType AS containerType,
        container.name AS containerName,
        door.ifcId AS doorId,
        door.name AS doorName,
        door.position_x,
        door.position_y,
        door.position_z

        # To find doors in walls using HAS_OPENING relationship
        MATCH (container)-[:HAS_OPENING]->(door)
        WHERE door.ifcType = 'IfcDoor'
        RETURN 
        container.ifcType AS containerType,
        container.name AS containerName,
        door.ifcId AS doorId,
        door.name AS doorName,
        door.position_x,
        door.position_y,
        door.position_z

        Question: {query}

        Cypher query:"""


                
        try:
            if AutoGen:
                cypher = self.llm.invoke(prompt).strip()
                # Clean up response
                cypher = cypher.replace("```cypher", "").replace("```", "").strip()
                # Remove any text before/after the query
                lines = cypher.split('\n')
                cypher_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
                cypher = '\n'.join(cypher_lines)
                
                print(f"🔧 Generated Cypher:\n{cypher}")
                return cypher
            
            else :
                saved_queries = self.load_saved_queries()
                prompt_catalog = "Please find in the catalog {saved_queries} the appropriate Cypher query for the following request: " + query
               
                cypher = self.llm.invoke(prompt_catalog).strip()

                return cypher

        except Exception as e:
            print(f"⚠️ Cypher generation failed: {e}")
            return None

    def ingest_neo4j_data(self, include_relationships: bool = True):
        """
        Comprehensive data ingestion that preserves ALL information.
        Creates multiple representations for optimal search.
        """
        print("\n📥 Ingesting complete Neo4j data...")
        
        # Strategy: Create multiple documents per node for different search purposes
        documents = []
        
        with self.driver.session() as sess:
            # Get all nodes with their complete schema and relationship information
            query = """
            MATCH (n)
            CALL (n) {
                OPTIONAL MATCH (n)-[r_out]->(m_out)
                RETURN collect(DISTINCT {
                    direction: 'OUT',
                    rel_type: type(r_out), 
                    target_id: elementId(m_out),
                    target_type: COALESCE(m_out.ifcType, head(labels(m_out))),
                    target_name: COALESCE(m_out.name, m_out.id, elementId(m_out)),
                    target_labels: labels(m_out)
                }) as outgoing_rels
            }
            CALL (n) {
                OPTIONAL MATCH (n)<-[r_in]-(m_in)
                RETURN collect(DISTINCT {
                    direction: 'IN',
                    rel_type: type(r_in), 
                    target_id: elementId(m_in),
                    target_type: COALESCE(m_in.ifcType, head(labels(m_in))),
                    target_name: COALESCE(m_in.name, m_in.id, elementId(m_in)),
                    target_labels: labels(m_in)
                }) as incoming_rels
            }
            WITH n, 
                 outgoing_rels + incoming_rels as relationships,
                 keys(n) as property_keys
            RETURN 
                 elementId(n) AS node_id,
                 labels(n) AS labels,
                 properties(n) AS properties,
                 relationships,
                 [k in property_keys | {key: k, value: n[k]}] as property_list
            """
            
            result = sess.run(query)
            
            for record in result:
                node_id = record["node_id"]
                labels = record["labels"]
                props = record["properties"]
                property_list = record["property_list"]
                # Build property types from actual values
                property_types = {pt['key']: type(pt['value']).__name__ for pt in property_list}
                relationships = [r for r in record["relationships"] if r.get('rel_type')]
                
                # Extract key properties with type awareness
                ifc_type = props.get('ifcType', '')
                name = props.get('name', props.get('id', f'Node_{node_id}'))
                
                # Add schema information to properties
                props['_schema'] = {
                    'labels': labels,
                    'property_types': property_types
                }
                
                # ========================================
                # Document 1: COMPLETE Properties List (PRIMARY)
                # ========================================
                # This is THE KEY document - it contains ALL properties explicitly
                prop_parts = [
                    f"=== ELEMENT PROPERTIES ===",
                    f"Type: {ifc_type}",
                    f"Name: {name}",
                    f"ID: {props.get('ifcId', node_id)}",
                    "",
                    "ALL PROPERTIES:"
                ]
                
                # Extract position first for special handling
                extracted_pos = None
                try:
                    if all(k in props for k in ['position_x', 'position_y', 'position_z']):
                        extracted_pos = [float(props['position_x']), float(props['position_y']), float(props['position_z'])]
                    else:
                        pos_field = props.get('position') or props.get('centroid') or props.get('coordinates')
                        if isinstance(pos_field, (list, tuple)) and 2 <= len(pos_field) <= 3:
                            extracted_pos = [float(x) for x in pos_field]
                        elif isinstance(pos_field, str):
                            import re
                            nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+", pos_field)
                            if 2 <= len(nums) <= 3:
                                extracted_pos = [float(x) for x in nums[:3]]
                except Exception:
                    extracted_pos = None
                
                # List ALL properties in detail
                for key, value in sorted(props.items()):
                    if key in ['_schema']:  # Skip internal
                        continue
                    
                    prop_type = property_types.get(key, type(value).__name__)
                    
                    # Format value for readability
                    if isinstance(value, (list, tuple)):
                        formatted_val = f"[{', '.join(map(str, value))}]"
                    elif isinstance(value, dict):
                        formatted_val = json.dumps(value, ensure_ascii=False)
                    elif isinstance(value, bool):
                        formatted_val = str(value).lower()
                    else:
                        formatted_val = str(value)
                    
                    prop_parts.append(f"  • {key}: {formatted_val} ({prop_type})")
                
                prop_content = "\n".join(prop_parts)
                
                # Prepare position-safe metadata
                pos_meta = {}
                if extracted_pos is not None:
                    try:
                        pos_meta = {
                            "position": json.dumps(extracted_pos),
                            "position_x": float(extracted_pos[0]),
                            "position_y": float(extracted_pos[1]),
                        }
                        if len(extracted_pos) >= 3:
                            pos_meta["position_z"] = float(extracted_pos[2])
                    except Exception:
                        pos_meta = {"position": json.dumps(extracted_pos)}
                
                doc_props = Document(
                    page_content=prop_content,
                    metadata={
                        "node_id": node_id,
                        **pos_meta,
                        "doc_type": "complete_properties",
                        "ifcType": ifc_type,
                        "name": name,
                        "labels": ", ".join(labels),
                        "all_properties": json.dumps(props),
                        "property_count": len([k for k in props.keys() if not k.startswith('_')]),
                    }
                )
                documents.append(doc_props)
                
                # ========================================
                # Document 2: Natural Language Description
                # ========================================
                # This is for semantic/conceptual search
                nl_parts = []
                
                if ifc_type:
                    nl_parts.append(f"This is a {ifc_type.replace('Ifc', '')} element")
                if name:
                    nl_parts.append(f"named '{name}'")
                
                # Add key properties in natural language
                property_descriptions = []
                
                for key, value in props.items():
                    if key in ['ifcType', 'name', 'id', '_schema', 'ifcId']:
                        continue
                    if key in ['position_x', 'position_y', 'position_z', 'position', 'centroid', 'coordinates']:
                        continue  # Handle position separately
                    
                    if value is None:
                        continue
                    
                    # Format based on type
                    if isinstance(value, (list, tuple)):
                        property_descriptions.append(f"{key}: {', '.join(map(str, value))}")
                    elif isinstance(value, dict):
                        property_descriptions.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
                    elif isinstance(value, bool):
                        property_descriptions.append(f"{key} is {'true' if value else 'false'}")
                    elif isinstance(value, (int, float)):
                        property_descriptions.append(f"{key}: {value}")
                    elif isinstance(value, str):
                        property_descriptions.append(f"{key}: {value}")
                
                # Add position if available
                if extracted_pos is not None:
                    property_descriptions.insert(0, f"located at position ({', '.join(map(str, extracted_pos))})")
                
                nl_parts.extend(property_descriptions)
                
                # Add relationships
                if relationships:
                    incoming_rels = []
                    outgoing_rels = []
                    for rel in relationships[:5]:
                        if rel['target_name']:
                            rel_desc = f"{rel['rel_type']} {rel['target_name']} ({rel['target_type']})"
                            if rel['direction'] == 'IN':
                                incoming_rels.append(rel_desc)
                            else:
                                outgoing_rels.append(rel_desc)
                    
                    if outgoing_rels:
                        nl_parts.append(f"It connects to: {', '.join(outgoing_rels)}")
                    if incoming_rels:
                        nl_parts.append(f"It is connected from: {', '.join(incoming_rels)}")
                
                nl_content = ". ".join(nl_parts) + "."
                
                doc_nl = Document(
                    page_content=nl_content,
                    metadata={
                        "node_id": node_id,
                        **pos_meta,
                        "doc_type": "natural_language",
                        "ifcType": ifc_type,
                        "name": name,
                        "labels": ", ".join(labels),
                        "all_properties": json.dumps(props),
                        "has_relationships": len(relationships) > 0,
                    }
                )
                documents.append(doc_nl)
                
                # ========================================
                # Document 3: Relationship-focused
                # ========================================
                if relationships:
                    rel_parts = [
                        f"=== RELATIONSHIPS FOR {name} ({ifc_type}) ===",
                        ""
                    ]
                    
                    outgoing = [r for r in relationships if r['direction'] == 'OUT']
                    incoming = [r for r in relationships if r['direction'] == 'IN']
                    
                    if outgoing:
                        rel_parts.append("OUTGOING:")
                        for rel in outgoing:
                            if rel['target_name']:
                                rel_parts.append(f"  → {rel['rel_type']} → {rel['target_name']} ({rel['target_type']})")
                    
                    if incoming:
                        rel_parts.append("")
                        rel_parts.append("INCOMING:")
                        for rel in incoming:
                            if rel['target_name']:
                                rel_parts.append(f"  ← {rel['rel_type']} ← {rel['target_name']} ({rel['target_type']})")
                    
                    rel_content = "\n".join(rel_parts)
                    
                    doc_rel = Document(
                        page_content=rel_content,
                        metadata={
                            "node_id": node_id,
                            **pos_meta,
                            "doc_type": "relationships",
                            "ifcType": ifc_type,
                            "name": name,
                            "labels": ", ".join(labels),
                            "relationship_count": len(relationships),
                            "all_properties": json.dumps(props),
                        }
                    )
                    documents.append(doc_rel)
        
        if documents:
            print(f"📝 Adding {len(documents)} documents to vector store...")
            print(f"   - Per node: ~2-3 documents (Complete Props + NL + Relationships)")
            
            # Batch insert for performance
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                self.vectorstore.add_documents(batch)
                print(f"   ✓ Processed {min(i + batch_size, len(documents))}/{len(documents)}")
            
            print("✅ Complete data ingestion finished!")
        else:
            print("⚠️ No documents found in Neo4j.")
        
        return len(documents)