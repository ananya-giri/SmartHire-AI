"""
Neuro-Symbolic Skill Knowledge Graph (corrected description, Gap-1 fix).

Ground truth about this implementation (docs previously described a
different graph):
  - NetworkX UNDIRECTED graph (nx.Graph), 70 nodes, 74 edges.
  - Edges are UNTYPED; weights are integers: 1 (most), 2, 3, 4, 5
    (heavier weights are cross-domain bridges that Dijkstra is meant
    to avoid as shortcuts).
  - Alias handling (synonyms such as k8s -> Kubernetes) lives in
    SKILL_ALIASES, not in graph edges.
  - Matching is BINARY per requirement: approved iff the shortest
    weighted path distance is 0, 1 or 2. There is no continuous
    1 - d/tau score in this implementation.
"""
import networkx as nx
from typing import Optional, Union, List
try:
    # pyrefly: ignore [missing-import]
    from crewai.tools import tool
except ImportError:  # Gap-3 reproducibility fix: deterministic tools must
    # be importable and testable without the full CrewAI stack installed.
    def tool(_name=None, *a, **k):
        def deco(f):
            return f
        return deco(_name) if callable(_name) else deco

# Common tech alias mappings for robust normalization
SKILL_ALIASES = {
    "reactjs": "React",
    "react.js": "React",
    "nextjs": "Next.js",
    "next": "Next.js",
    "vuejs": "Vue",
    "vue.js": "Vue",
    "angularjs": "Angular",
    "nodejs": "Node.js",
    "node": "Node.js",
    "expressjs": "Express",
    "express.js": "Express",
    "fast-api": "FastAPI",
    "ts": "TypeScript",
    "typescript": "TypeScript",
    "js": "JavaScript",
    "javascript": "JavaScript",
    "py": "Python",
    "python3": "Python",
    "golang": "Go",
    "k8s": "Kubernetes",
    "k8": "Kubernetes",
    "docker-compose": "Docker",
    "amazon web services": "AWS",
    "google cloud": "GCP",
    "google cloud platform": "GCP",
    "microsoft azure": "Azure",
    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "mongo": "MongoDB",
    "mongodb": "MongoDB",
    "sklearn": "Scikit-Learn",
    "scikit_learn": "Scikit-Learn",
    "scikitlearn": "Scikit-Learn",
    "pytorch": "PyTorch",
    "tf": "TensorFlow",
    "tensorflow": "TensorFlow",
    "ml": "Machine Learning",
    "ai": "Artificial Intelligence",
    "genai": "Generative AI",
    "gen-ai": "Generative AI",
    "generative ai": "Generative AI",
    "llm": "LLMs",
    "llms": "LLMs",
    "ci/cd": "CI/CD",
    "cicd": "CI/CD",
    "github actions": "GitHub Actions",
    "gh actions": "GitHub Actions",
    "restful": "REST",
    "rest api": "REST",
    "rest apis": "REST",
    "graph-ql": "GraphQL",
    "graphql": "GraphQL"
}

def build_tech_graph() -> nx.Graph:
    G = nx.Graph()
    
    # Domains & Frontend
    G.add_edge("Frontend", "JavaScript", weight=1)
    G.add_edge("Frontend", "TypeScript", weight=1)
    G.add_edge("JavaScript", "TypeScript", weight=1)
    G.add_edge("JavaScript", "React", weight=1)
    G.add_edge("TypeScript", "React", weight=1)
    G.add_edge("React", "Next.js", weight=1)
    G.add_edge("JavaScript", "Vue", weight=1)
    G.add_edge("Vue", "Nuxt", weight=1)
    G.add_edge("TypeScript", "Angular", weight=1)
    G.add_edge("Frontend", "HTML5", weight=2)
    G.add_edge("Frontend", "CSS3", weight=2)
    G.add_edge("CSS3", "TailwindCSS", weight=1)
    G.add_edge("React", "Redux", weight=1)
    G.add_edge("React", "Zustand", weight=1)
    
    # Backend & Languages
    G.add_edge("Backend", "Python", weight=1)
    G.add_edge("Backend", "Node.js", weight=1)
    G.add_edge("Backend", "Java", weight=1)
    G.add_edge("Backend", "Go", weight=1)
    G.add_edge("Backend", "C++", weight=1)
    G.add_edge("Backend", "Rust", weight=1)
    G.add_edge("Python", "Django", weight=1)
    G.add_edge("Python", "FastAPI", weight=1)
    G.add_edge("Python", "Flask", weight=1)
    G.add_edge("Node.js", "Express", weight=1)
    G.add_edge("Node.js", "NestJS", weight=1)
    G.add_edge("Java", "Spring Boot", weight=1)
    G.add_edge("Backend", "REST", weight=1)
    G.add_edge("Backend", "GraphQL", weight=1)
    G.add_edge("Backend", "gRPC", weight=1)
    G.add_edge("Backend", "Microservices", weight=1)
    
    # Cloud, Infrastructure & DevOps
    G.add_edge("Cloud", "AWS", weight=1)
    G.add_edge("Cloud", "GCP", weight=1)
    G.add_edge("Cloud", "Azure", weight=1)
    G.add_edge("AWS", "EC2", weight=1)
    G.add_edge("AWS", "S3", weight=1)
    G.add_edge("AWS", "Lambda", weight=1)
    G.add_edge("DevOps", "Docker", weight=1)
    G.add_edge("DevOps", "Kubernetes", weight=1)
    G.add_edge("Docker", "Kubernetes", weight=1)
    G.add_edge("DevOps", "Terraform", weight=1)
    G.add_edge("DevOps", "CI/CD", weight=1)
    G.add_edge("CI/CD", "GitHub Actions", weight=1)
    G.add_edge("CI/CD", "Jenkins", weight=1)
    G.add_edge("DevOps", "Linux", weight=1)
    G.add_edge("DevOps", "Git", weight=1)
    
    # Databases & Storage
    G.add_edge("Database", "SQL", weight=1)
    G.add_edge("SQL", "PostgreSQL", weight=1)
    G.add_edge("SQL", "MySQL", weight=1)
    G.add_edge("Database", "MongoDB", weight=1)
    G.add_edge("Database", "Redis", weight=1)
    G.add_edge("Database", "Snowflake", weight=1)
    G.add_edge("Backend", "Database", weight=2)
    G.add_edge("Backend", "Kafka", weight=1)
    G.add_edge("Backend", "RabbitMQ", weight=1)
    
    # Data & Machine Learning / AI
    G.add_edge("Python", "Machine Learning", weight=1)
    G.add_edge("Artificial Intelligence", "Machine Learning", weight=1)
    G.add_edge("Machine Learning", "Deep Learning", weight=1)
    G.add_edge("Machine Learning", "PyTorch", weight=1)
    G.add_edge("Machine Learning", "TensorFlow", weight=1)
    G.add_edge("Machine Learning", "Scikit-Learn", weight=1)
    G.add_edge("Machine Learning", "Pandas", weight=1)
    G.add_edge("Machine Learning", "NumPy", weight=1)
    G.add_edge("Deep Learning", "NLP", weight=1)
    G.add_edge("Deep Learning", "Computer Vision", weight=1)
    G.add_edge("NLP", "LLMs", weight=1)
    G.add_edge("LLMs", "Generative AI", weight=1)
    G.add_edge("Data Engineering", "SQL", weight=1)
    G.add_edge("Data Engineering", "Spark", weight=1)
    G.add_edge("Data Engineering", "Airflow", weight=1)
    G.add_edge("Data Engineering", "Python", weight=1)
    
    # Cross-domain bridges (weighted heavily to prevent generic domain shortcuts)
    G.add_edge("Frontend", "Backend", weight=5)
    G.add_edge("Backend", "Cloud", weight=4)
    G.add_edge("Cloud", "DevOps", weight=3)
    G.add_edge("Backend", "Data Engineering", weight=4)

    return G

# Singleton graph instance initialized at module load
_GLOBAL_TECH_GRAPH = build_tech_graph()

def _normalize_skill_name(skill_str: str, graph_nodes: set) -> str:
    """Normalizes skill string using alias mapping and case-insensitive matching."""
    s = skill_str.strip()
    s_lower = s.lower().replace("-", " ").replace("_", " ")
    
    # Check direct aliases
    if s.lower() in SKILL_ALIASES:
        return SKILL_ALIASES[s.lower()]
    if s_lower in SKILL_ALIASES:
        return SKILL_ALIASES[s_lower]
        
    # Check exact case match with graph
    for node in graph_nodes:
        if node.lower() == s.lower() or node.lower() == s_lower:
            return node
            
    return s

def evaluate_skill_match(required_skill: str, candidate_skills: Union[str, List[str]]) -> dict:
    """
    Evaluates skill match using weighted shortest path on the curated tech graph.
    Returns structured decision dictionary with:
      - match_type: EXACT | ALIAS | DIRECT_SUBSUMPTION | TRANSFERABLE | DISTANT_PROXIMITY | GAP | UNMAPPED
      - weighted_distance: float (0.0 for exact/alias, inf for gap/no path)
      - closest_skill: str
      - is_approved: bool (True only for exact, alias, and distance <= 2)
      - details: str
    """
    if not required_skill or not str(required_skill).strip():
        return {
            "match_type": "ERROR",
            "weighted_distance": float('inf'),
            "closest_skill": None,
            "is_approved": False,
            "details": "Error: required_skill cannot be empty."
        }

    G = _GLOBAL_TECH_GRAPH
    graph_nodes = set(G.nodes)
    
    req_clean = str(required_skill).strip()
    req_mapped = _normalize_skill_name(req_clean, graph_nodes)
    
    # Parse candidate skills
    if not candidate_skills:
        candidate_skills_list = []
    elif isinstance(candidate_skills, list):
        candidate_skills_list = [str(x).strip() for x in candidate_skills if str(x).strip()]
    else:
        candidate_skills_list = [s.strip() for s in str(candidate_skills).replace(";", ",").replace("\n", ",").split(",") if s.strip()]
    
    if not candidate_skills_list:
        return {
            "match_type": "GAP",
            "weighted_distance": float('inf'),
            "closest_skill": None,
            "is_approved": False,
            "details": f"No candidate skills provided. Cannot match '{req_clean}'."
        }

    # 1. Exact textual match check
    for c_skill in candidate_skills_list:
        if c_skill.lower() == req_clean.lower():
            return {
                "match_type": "EXACT",
                "weighted_distance": 0.0,
                "closest_skill": req_clean,
                "is_approved": True,
                "details": f"Exact Explicit Match found for '{req_clean}'."
            }

    # 2. Alias resolution check
    for c_skill in candidate_skills_list:
        c_mapped = _normalize_skill_name(c_skill, graph_nodes)
        if c_mapped.lower() == req_mapped.lower() and c_mapped != c_skill:
            return {
                "match_type": "ALIAS",
                "weighted_distance": 0.0,
                "closest_skill": c_skill,
                "is_approved": True,
                "details": f"Alias resolved: '{c_skill}' maps directly to canonical '{req_mapped}'."
            }

    # 3. Graph availability check
    if req_mapped not in G:
        return {
            "match_type": "UNMAPPED",
            "weighted_distance": float('inf'),
            "closest_skill": None,
            "is_approved": False,
            "details": f"Skill '{req_clean}' is specialized or not mapped in standard generic graph. LLM must use native contextual reasoning."
        }
    
    closest_skill = None
    min_dist = float('inf')
    
    for skill in candidate_skills_list:
        skill_mapped = _normalize_skill_name(skill, graph_nodes)
        if skill_mapped in G:
            try:
                # Use Dijkstra weighted shortest path length
                dist = nx.shortest_path_length(G, source=req_mapped, target=skill_mapped, weight='weight')
                if dist < min_dist:
                    min_dist = dist
                    closest_skill = skill_mapped
            except nx.NetworkXNoPath:
                pass
                
    if min_dist == 0:
        return {
            "match_type": "EXACT",
            "weighted_distance": 0.0,
            "closest_skill": closest_skill,
            "is_approved": True,
            "details": f"Exact Explicit Match found for '{req_clean}'."
        }
    elif min_dist == 1:
        return {
            "match_type": "DIRECT_SUBSUMPTION",
            "weighted_distance": min_dist,
            "closest_skill": closest_skill,
            "is_approved": True,
            "details": f"Latent Neuro-Symbolic match found: Candidate lacks explicit '{req_clean}', but has '{closest_skill}' which is directly coupled in the knowledge graph (Weighted Distance: {min_dist}). Approve skill."
        }
    elif min_dist == 2:
        return {
            "match_type": "TRANSFERABLE",
            "weighted_distance": min_dist,
            "closest_skill": closest_skill,
            "is_approved": True,
            "details": f"Transferable skill match found: Candidate has '{closest_skill}', an adjacent stack technology to '{req_clean}' (Weighted Distance: {min_dist}). Approve skill."
        }
    elif min_dist == 3:
        return {
            "match_type": "DISTANT_PROXIMITY",
            "weighted_distance": min_dist,
            "closest_skill": closest_skill,
            "is_approved": False,
            "details": f"Moderate Domain proximity found between '{req_clean}' and candidate's skill '{closest_skill}' (Weighted Distance: {min_dist}). Flagged for technical interview assessment; not automatically approved."
        }
    else:
        return {
            "match_type": "GAP",
            "weighted_distance": min_dist,
            "closest_skill": closest_skill,
            "is_approved": False,
            "details": f"No close semantic connection found in the skill network for '{req_clean}' (Weighted Distance: {min_dist}). Flag as a gap."
        }

@tool("semantic_skill_graph_matcher")
def semantic_skill_graph_matcher(required_skill: str, candidate_skills_csv: str) -> str:
    """Useful to mathematically measure the distance between a required job skill and a candidate's actual skills using a Neuro-Symbolic Knowledge Graph.
    Pass the required_skill and a comma-separated list of candidate_skills. 
    It returns if there is a latent mathematical semantic connection."""
    res = evaluate_skill_match(required_skill, candidate_skills_csv)
    return res["details"]

def get_kg_tool():
    return semantic_skill_graph_matcher



def get_graph_stats() -> dict:
    """Verified structural statistics of the skill graph."""
    from collections import Counter
    G = _GLOBAL_TECH_GRAPH
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "directed": G.is_directed(),
        "weight_distribution": dict(Counter(d.get("weight", 1) for _, _, d in G.edges(data=True))),
        "aliases": len(SKILL_ALIASES),
    }
