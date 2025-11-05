import json
import math
from typing import Optional, Dict, Any, List
import ollama
from langchain_ollama import OllamaLLM
from neo4j import GraphDatabase


class cypher_query_utils:
    def __init__(self, llm, neo4j_uri: str, neo4j_auth: tuple, vectorstore: Any):
        
        self.llm = OllamaLLM(model="qwen2.5vl:3b") if llm is None else llm
        self.neo4j_uri = neo4j_uri
        self.neo4j_auth = neo4j_auth
        self.vectorstore = vectorstore
        object.__setattr__(self, 'driver', GraphDatabase.driver(neo4j_uri, auth=neo4j_auth))

    def load_saved_queries(self, path: str = "cypher_queries.json") -> List[Dict[str, Any]]:
        """Load saved Cypher queries from a JSON file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("Saved queries file must contain a list of query objects.")
            return data
        except FileNotFoundError:
            print(f"⚠️ File not found: {path}")
            return []
        except json.JSONDecodeError as e:
            print(f"⚠️ Failed to parse JSON: {e}")
            return []

    def _summarize_catalog_for_prompt(self, catalog: List[Dict[str, Any]]) -> str:
        """Compact, LLM-friendly catalog summary (names, descriptions, keywords only)."""
        lines = []
        for i, q in enumerate(catalog, start=1):
            name = q.get("name", "")
            desc = (q.get("description", "") or "").replace("\n", " ").strip()
            kws  = ", ".join(q.get("keywords", []) or [])
            lines.append(f'{i}. name="{name}" | desc="{desc}" | keywords=[{kws}]')
        return "\n".join(lines)

    def _safe_extract_json(self, text: str) -> Dict[str, Any]:
        """Extract the first {...} JSON object from LLM output and parse it."""
        start = text.find("{")
        end   = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in LLM output.")
        return json.loads(text[start:end+1])
    
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


    def _simple_keyword_score(self, user_query: str, q: Dict[str, Any]) -> float:
        uq = user_query.lower()
        kws = [str(k).lower() for k in (q.get("keywords") or [])]
        hits = sum(1 for k in kws if k and k in uq)
        # small boost for description token overlap
        def toks(s): 
            return set("".join(c.lower() if c.isalnum() else " " for c in s).split())
        overlap = len(toks(user_query) & toks(q.get("description", "")))
        return hits * 2.0 + overlap * 0.25

    def _best_fallback_name(self, user_query: str, catalog: List[Dict[str, Any]], min_score: float = 1.0) -> Optional[str]:
        best, best_score = None, -math.inf
        for q in catalog:
            s = self._simple_keyword_score(user_query, q)
            if s > best_score:
                best, best_score = q.get("name"), s
        return best if best_score >= min_score else None

    def _sanitize_readonly_cypher(self, cypher: str, default_limit: int = 1000) -> Optional[str]:
        """One-statement, read-only, enforce LIMIT if returning rows."""
        c = cypher.strip().strip("`")
        # collapse code fences if present
        c = c.replace("```cypher", "").replace("```", "").strip()
        # one statement only
        parts = [p.strip() for p in c.split(";") if p.strip()]
        if len(parts) != 1:
            return None
        c = parts[0]
        UC = c.upper()
        forbidden = ("CREATE", "MERGE", "SET ", "DELETE", "DETACH", "REMOVE", "DROP", "ALTER", "LOAD CSV")
        if any(f in UC for f in forbidden):
            return None
        if not (UC.startswith("MATCH") or UC.startswith("CALL") or UC.startswith("WITH") or UC.startswith("UNWIND") or UC.startswith("RETURN")):
            return None
        # add LIMIT if there's a RETURN but no LIMIT
        if " RETURN " in f" {UC} " and " LIMIT " not in UC:
            c = f"{c}\nLIMIT {default_limit}"
        return c

    def _index_catalog(self, catalog: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        idx = {}
        for q in catalog:
            name = q.get("name")
            if not name:
                continue
            if name in idx:
                raise ValueError(f"Duplicate saved query name: {name}")
            idx[name] = q
        return idx

    def generate_cypher_from_query(self, query: str, AutoGen: bool = False) -> Optional[str]:
        """
        Generate or retrieve a Cypher query for the given natural-language user query.

        Modes:
        - AutoGen=False (default): Use LLM to select a pre-saved query from catalog (saved_queries.json)
                                    based on description & keywords.
        - AutoGen=True:  Automatically generate a Cypher query from the IFC schema and instructions.

        If the catalog route fails, it automatically falls back to AutoGen.
        """
        try:
            # ───────────────────────────────
            # 1️⃣ Direct generation path (AutoGen=True)
            # ───────────────────────────────
            if AutoGen:
                schema = self.get_complete_graph_schema()
                prompt = f"""
    You are an expert in BIM (Building Information Modeling) and Neo4j graph databases.

    I have imported an IFC building model into Neo4j with this schema:

    {schema}

    Rules for query generation:
    - Generate **exactly one** valid Neo4j Cypher query.
    - Query must be read-only (MATCH / OPTIONAL MATCH / WHERE / WITH / RETURN / ORDER BY / LIMIT only).
    - Do NOT use CREATE, MERGE, SET, DELETE, or APOC write procedures.
    - If a LIMIT is not provided, append LIMIT 200.
    - If geometry may be missing, handle it safely (e.g. WHERE exists(node.position_x)).

    Here are example patterns you can reference (not required to reuse verbatim):

    MATCH p = ()-[:CONTAINS]->() RETURN p LIMIT 600;
    MATCH (n:IfcDoor) RETURN n LIMIT 100;
    MATCH (container)-[:HAS_OPENING]->(window:IfcWindow)
    RETURN container.ifcType, window.ifcId, window.position_x, window.position_y, window.position_z LIMIT 200;

    Now, write the appropriate Cypher query for the following request:

    User question: "{query}"

    Return ONLY the Cypher query (no explanations).
                """

                raw = self.llm.invoke(prompt).strip()
                cypher = self._sanitize_readonly_cypher(raw)
                if not cypher:
                    print("⚠️ Auto-generated Cypher failed sanitation.")
                    return None

                print(f"🤖 Auto-generated Cypher:\n{cypher}")
                return cypher

            # ───────────────────────────────
            # 2️⃣ Catalog routing path (AutoGen=False)
            # ───────────────────────────────
            catalog = self.load_saved_queries()
            index   = self._index_catalog(catalog)

            # Summarize catalog for LLM
            catalog_summary = self._summarize_catalog_for_prompt(catalog)
            router_prompt = f"""
    You are a precise routing assistant.

    From the following catalog of saved Cypher queries, pick the ONE query NAME that best answers the user's question.
    If none fit, respond with name=null.

    Return STRICT JSON only:
    {{"name": <string|null>, "confidence": <0..1>, "matched_keywords": [<strings>], "reason": <string>}}

    Catalog:
    {catalog_summary}

    User query: "{query}"

    JSON:
            """

            resp = self.llm.invoke(router_prompt)
            text = resp.strip() if isinstance(resp, str) else str(resp).strip()
            data = self._safe_extract_json(text)

            picked_name = data.get("name")
            confidence  = float(data.get("confidence", 0.0) or 0.0)

            if isinstance(picked_name, str) and picked_name in index and confidence >= 0.55:
                cypher = index[picked_name]["cypher"]
                cypher = self._sanitize_readonly_cypher(cypher) or cypher
                print(f"✅ Router selected saved query: {picked_name} (confidence={confidence:.2f})")
                return cypher

            # Fallback: try simple keyword match
            fb_name = self._best_fallback_name(query, catalog)
            if fb_name and fb_name in index:
                cypher = index[fb_name]["cypher"]
                cypher = self._sanitize_readonly_cypher(cypher) or cypher
                print(f"ℹ️ Router fallback matched: {fb_name}")
                return cypher

            # Nothing matched — generate automatically
            print("⚙️ No suitable saved query found, switching to AutoGen.")
            return self.generate_cypher_from_query(query, AutoGen=True)

        except Exception as e:
            print(f"⚠️ Cypher generation failed: {e}")
            return None
