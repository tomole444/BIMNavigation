"""Runner for the Neo4jGraphHybridRetriever.

Usage: run this file from the `src/BIM_Models` directory:

    python3 runner.py

The runner sets up embeddings, the vectorstore and retriever, optionally ingests data
and runs a small set of queries. This keeps the retriever implementation separate
from execution logic.
"""
import argparse
from typing import List
import json

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from langchain_reteriver import Neo4jGraphHybridRetriever, create_retrieval_chain


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--no-ingest", action="store_true", help="Skip ingestion (use existing vector DB)")
    p.add_argument("--queries", nargs="*", help="Queries to run (overrides default)")
    p.add_argument("--k", type=int, default=20, help="k for vector search")
    return p.parse_args()


def main():
    args = parse_args()

    # Configuration (move these into environment variables or a config file if desired)
    NEO4J_URI = "neo4j://127.0.0.1:7687"
    NEO4J_AUTH = ("neo4j", "kg_bim_cranehall")
    EMBEDDING_MODEL = "qwen2.5-coder:7b"
    LLM_MODEL = "llama3.2-vision:11b"

    print("🚀 Initializing GraphRAG components...")

    # Create embeddings + vector store
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        collection_name="bim_complete_v2",
        embedding_function=embeddings,
        persist_directory="./chroma_db_complete",
    )

    retriever = Neo4jGraphHybridRetriever(
        vectorstore=vectorstore,
        neo4j_uri=NEO4J_URI,
        neo4j_auth=NEO4J_AUTH,
        llm_model=LLM_MODEL,
        max_traverse_depth=2,
        k=args.k,
    )

    if not args.no_ingest:
        print("\n=== INGESTION ===")
        doc_count = retriever.ingest_neo4j_data()
        print(f"Ingested {doc_count} documents")

    qa_chain = create_retrieval_chain(retriever)

    queries: List[str] = args.queries or [
        "can you tell me all the doors and their bounding boxes and positions",
    ]

    print("\n=== QUERY PHASE ===")
    for q in queries:
        print(f"\n--- Query: {q}")
        try:
            res = qa_chain.invoke({"query": q})
            print("Answer:\n", json.dumps(res, indent=2, default=str))
        except Exception as e:
            print("Query failed:", e)

    retriever.close()


if __name__ == "__main__":
    main()
