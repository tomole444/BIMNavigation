"""
GraphRAG Retriever with Complete Data Preservation
---------------------------------------------------
Preserves ALL Neo4j data and makes it fully searchable
"""

from typing import List, Dict, Any, Optional
import json
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from neo4j import GraphDatabase
from langchain_utils import retrieverUtils
from cypher_query_utils import cypher_query_utils
import traceback



class Neo4jGraphHybridRetriever(BaseRetriever):
    """
    Complete retriever that preserves ALL Neo4j data.
    Uses multiple strategies for different query types.
    """
    
    vectorstore: Chroma
    neo4j_uri: str
    neo4j_auth: tuple
    llm: OllamaLLM
    max_traverse_depth: int = 1
    k: int = 20
    driver: Any = None
    
    class Config:
        """Pydantic config for arbitrary types."""
        arbitrary_types_allowed = True
    
    def __init__(
        self,
        vectorstore: Chroma,
        neo4j_uri: str,
        neo4j_auth: tuple,
        llm_model: str,
        max_traverse_depth: int = 1,
        k: int = 20,
    ):
        """Initialize the hybrid retriever."""
        super().__init__(
            vectorstore=vectorstore,
            neo4j_uri=neo4j_uri,
            neo4j_auth=neo4j_auth,
            llm=OllamaLLM(model=llm_model,
                          temperature=0),
            max_traverse_depth=max_traverse_depth,
            k=k,
            driver=None,
        )
        object.__setattr__(self, 'driver', GraphDatabase.driver(neo4j_uri, auth=neo4j_auth))
        object.__setattr__(self, 'retriever_utils', retrieverUtils(neo4j_uri, neo4j_auth, vectorstore))
        #object.__setattr__(self, 'cypher_llm', True)
        object.__setattr__(self, 'cypher_utils',
            cypher_query_utils(
                llm=OllamaLLM(model=llm_model),   
                neo4j_uri=neo4j_uri,
                neo4j_auth=neo4j_auth,
                vectorstore=vectorstore
            )
        )    
   
    

    
    def query_neo4j_direct(self, cypher: str, params: Dict = None) -> List[Dict]:
        """
        Execute arbitrary Cypher queries directly.
        This preserves ALL data without conversion.
        """
        try:
            with self.driver.session() as sess:
                result = sess.run(cypher, params or {})
                records = []
                for record in result:
                    # Convert Neo4j record to dict, preserving all types
                    record_dict = {}
                    for key in record.keys():
                        value = record[key]
                        # Handle Neo4j types
                        if hasattr(value, '__dict__'):
                            record_dict[key] = dict(value)
                        else:
                            record_dict[key] = value
                    records.append(record_dict)
                return records
        except Exception as e:
            print(f"⚠️ Cypher query failed: {e}")
            return []
    
    
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun = None,
    ) -> List[Document]:
        """
        Multi-strategy retrieval:
        1. Direct Cypher (for structured queries)
        2. Vector search (for semantic queries)
        3. Graph traversal (for relationships)
        """
        print(f"\n{'='*70}")
        print(f"🔍 Query: '{query}'")
        print('='*70)
        
        all_docs = []
        
        # ========================================
        # Strategy 1: Direct Cypher Execution
        # ========================================
        # Best for: counts, filters, structured queries

        print("\n🧠 Generating Cypher via LLM...")
        cypher = self.cypher_utils.generate_cypher_from_query(query, AutoGen=False)
        print (f"📝 Generated Cypher:\n{cypher}")
        

        if cypher:
            results = self.query_neo4j_direct(cypher)
            if results:
                print(f"✅ Direct Cypher returned {len(results)} results")
                
                # DEBUG: show raw direct results (first few)
                print("[DEBUG] Raw direct Cypher results sample:")
                for rr in results[:5]:
                    try:
                        print(json.dumps(rr, default=str, ensure_ascii=False))
                    except Exception:
                        print(str(rr))
                print("[END DEBUG]")

                for i, result in enumerate(results[:100]):  # Limit display
                    # Create rich document from result
                    content_parts = []
                    metadata = {"source": "neo4j_direct", "result_index": i}

                    # temporary holder for potential position components
                    pos_vals = {}

                    for key, value in result.items():
                        lk = str(key).lower()
                        # Node or relationship object
                        if isinstance(value, dict):
                            if 'ifcType' in value:
                                # prefer name + type
                                content_parts.append(f"{value.get('name', 'Unnamed')} ({value.get('ifcType')})")
                            metadata[key] = json.dumps(value)
                            # also extract any positional props inside node dict
                            if isinstance(value, dict):
                                for pk in ('position_x', 'position_y', 'position_z'):
                                    if pk in value and value[pk] is not None:
                                        try:
                                            metadata[pk] = float(value[pk])
                                            pos_vals[pk] = float(value[pk])
                                        except Exception:
                                            metadata[pk] = str(value[pk])
                        else:
                            # Primitive value: check if it's a position component or position container
                            if any(q in lk for q in ['position_x', 'positiony', 'position_x'.replace('_','')]):
                                # direct position components
                                try:
                                    metadata[lk] = float(value) if value is not None else None
                                    pos_vals[lk] = float(value) if value is not None else None
                                except Exception:
                                    metadata[lk] = str(value)
                                content_parts.append(f"{key}: {value}")
                            elif 'position' in lk and isinstance(value, (list, tuple)):
                                # whole position list
                                try:
                                    coords = [float(x) for x in value]
                                    metadata['position'] = json.dumps(coords)
                                    if len(coords) > 0:
                                        metadata['position_x'] = coords[0]
                                    if len(coords) > 1:
                                        metadata['position_y'] = coords[1]
                                    if len(coords) > 2:
                                        metadata['position_z'] = coords[2]
                                    content_parts.append(f"position: ({', '.join(map(str, coords))})")
                                except Exception:
                                    metadata[key] = str(value)
                            else:
                                # fallback
                                content_parts.append(f"{key}: {value}")
                                metadata[key] = str(value) if value is not None else ""

                    # if we collected separate position components, add consolidated position to content/metadata
                    if pos_vals:
                        try:
                            px = pos_vals.get('position_x') or pos_vals.get('positionx') or pos_vals.get('position_x')
                            py = pos_vals.get('position_y') or pos_vals.get('positiony') or pos_vals.get('position_y')
                            pz = pos_vals.get('position_z') or pos_vals.get('positionz') or None
                            coords = []
                            if px is not None:
                                coords.append(float(px))
                            if py is not None:
                                coords.append(float(py))
                            if pz is not None:
                                coords.append(float(pz))
                            if coords:
                                metadata['position'] = json.dumps(coords)
                                metadata['position_x'] = coords[0]
                                if len(coords) > 1:
                                    metadata['position_y'] = coords[1]
                                if len(coords) > 2:
                                    metadata['position_z'] = coords[2]
                                content_parts.append(f"Position: ({', '.join(map(str, coords))})")
                        except Exception:
                            pass

                    content = " | ".join(content_parts)
                    doc = Document(page_content=content, metadata=metadata)
                    all_docs.append(doc)

                # DEBUG: show created documents from direct Cypher
                print("[DEBUG] Created Documents from direct Cypher (sample):")
                for d in all_docs[:min(10, len(all_docs))]:
                    try:
                        print(f"- page_content: {d.page_content}")
                        print(f"  metadata: {json.dumps(d.metadata, default=str, ensure_ascii=False)}")
                    except Exception:
                        print(f"  metadata: {d.metadata}")
                print("[END DEBUG]")
        
       
        return all_docs
    
    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
    
    def __del__(self):
        """Cleanup."""
        try:
            self.close()
        except:
            pass


def create_retrieval_chain(retriever: Neo4jGraphHybridRetriever) -> RetrievalQA:
    """Create QA chain."""
    return RetrievalQA.from_chain_type(
        llm=retriever.llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        verbose=True,
    )


if __name__ == "__main__":
    # Configuration
    NEO4J_URI = "neo4j://127.0.0.1:7687"
    NEO4J_AUTH = ("neo4j", "riedelbau_model")
    EMBEDDING_MODEL = "qwen3-embedding:latest"
    LLM_MODEL = "llama3.2-vision:11b"
    
    print("🚀 Initializing Complete GraphRAG Retriever...")
    
    # Initialize
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        collection_name="bim_riedel_bau",  # New collection name
        embedding_function=embeddings,
        persist_directory="./chroma_db_complete"
    )
    
    retriever = Neo4jGraphHybridRetriever(
        vectorstore=vectorstore,
        neo4j_uri=NEO4J_URI,
        neo4j_auth=NEO4J_AUTH,
        llm_model=LLM_MODEL,
        max_traverse_depth=10,
        k=10,
    )
    
    #Ingest data (comment out after first run)
    # print("\n" + "="*70)
    # print("INGESTION PHASE")
    # print("="*70)

    # langchain_utils = retrieverUtils(NEO4J_URI, NEO4J_AUTH, vectorstore)

    # doc_count = langchain_utils.ingest_neo4j_data()
    # print(f"\n✅ Ingested {doc_count} documents")
    
    # if doc_count == 0:
    #     print("\n⚠️ No data found in Neo4j")
    #     retriever.close()
    #     exit()
    
    # Create QA chain
    qa_chain = create_retrieval_chain(retriever)
    
    # Test queries
    print("\n" + "="*70)
    print("QUERY PHASE")
    print("="*70)
    
    test_queries = [
        " can you list all windows in the building along with their positions ?"
    ]

    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"❓ Question: {query}")
        print('='*70)
        
        try:
            result = qa_chain.invoke({"query": query})
            print(f"\n💡 Answer:\n{result['result']}")
            
            if result.get('source_documents'):
                print(f"\n📚 Used {len(result['source_documents'])} sources")
                
        except Exception as e:
            print(f"\n❌ Error: {e}")
            traceback.print_exc()
    
    #retriever.close()
    print("\n✅ Complete!")