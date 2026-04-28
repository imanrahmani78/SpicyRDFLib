"""
SpicyRDFLib — Real-world ontology test
=======================================
Loads the Rio Tinto CRA knowledge graph (ontology + instances) and
runs semantic SPARQL queries that would be impossible with exact matching.
"""

import sys
import time
sys.path.insert(0, ".")

from rdflib import Graph
from spicy import SpicyGraph
from spicy.embedder import _get_model

ONTOLOGY  = "Ontology/RioTintoCRA_NonInsV4.ttl"
INSTANCES = "Ontology/RioTintoCRA_InsV4.1.ttl"

PREFIX = """
PREFIX rtcra: <http://riotinto.cra.analytical.ontology#>
PREFIX rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs:  <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd:   <http://www.w3.org/2001/XMLSchema#>
PREFIX ext:   <https://spicyrdflib.dev/fn#>
"""

def section(title):
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print('─' * 70)

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading ontology + knowledge graph...")
t0 = time.perf_counter()
g = SpicyGraph()
g.parse(ONTOLOGY,  format="turtle")
g.parse(INSTANCES, format="turtle")
load_time = time.perf_counter() - t0
print(f"  {len(g):,} triples loaded in {load_time:.1f}s")

print("\nBuilding vector index (batched)...")
t0 = time.perf_counter()
g.enable_vector_search()
idx_time = time.perf_counter() - t0
model = _get_model()
print(f"  Device: {model.device}  |  Index built in {idx_time:.1f}s")

# ── Q1: Semantic search over incident descriptions ─────────────────────────
section("Q1 · Scenarios semantically related to 'structural damage to elevated equipment'")

r = list(g.query(PREFIX + """
SELECT ?scenario ?peril ?summary ?score WHERE {
  ?scenario rtcra:lossScenarioPeril  ?peril ;
            rtcra:incidentSummary    ?summary .
  BIND(ext:cosine_object(?summary, "structural damage to elevated equipment or conveyor") AS ?score)
  FILTER(?score > 0.5)
}
ORDER BY DESC(?score)
LIMIT 8
"""))
print(f"  {len(r)} results\n")
for row in r:
    print(f"  score={float(row.score):.3f}  [{row.peril}]")
    print(f"         {str(row.summary)[:100]}")


# ── Q2: Find fire and explosion risk scenarios ─────────────────────────────
section("Q2 · Fire or explosion risk scenarios (semantic, not substring)")

r = list(g.query(PREFIX + """
SELECT ?scenario ?peril ?summary ?score WHERE {
  ?scenario rtcra:lossScenarioPeril  ?peril ;
            rtcra:incidentSummary    ?summary .
  BIND(ext:cosine_object(?peril, "fire ignition explosion") AS ?score)
  FILTER(?score > 0.45)
}
ORDER BY DESC(?score)
LIMIT 8
"""))
print(f"  {len(r)} results\n")
for row in r:
    print(f"  score={float(row.score):.3f}  [{row.peril}]  {str(row.summary)[:80]}")


# ── Q3: Recommendations about equipment maintenance ────────────────────────
section("Q3 · Recommendations semantically related to 'equipment maintenance'")

r = list(g.query(PREFIX + """
SELECT ?rec ?summary ?status ?priority ?score WHERE {
  ?rec rdf:type                       rtcra:Recommendation ;
       rtcra:recommendationSummary    ?summary ;
       rtcra:recommendationStatus     ?status ;
       rtcra:recommendationPriority   ?priority .
  BIND(ext:cosine_object(?summary, "preventive maintenance inspection of equipment") AS ?score)
  FILTER(?score > 0.55)
}
ORDER BY DESC(?score)
LIMIT 6
"""))
print(f"  {len(r)} results\n")
for row in r:
    print(f"  score={float(row.score):.3f}  [{row.priority}]  status=[{row.status}]")
    print(f"         {str(row.summary)[:110]}")


# ── Q4: Combine semantic risk search with structural (operation + score) ────
section("Q4 · High-scoring collapse/subsidence risks + which operation they belong to")

r = list(g.query(PREFIX + """
SELECT ?operation ?scenario ?peril ?summary ?score WHERE {
  ?scenario rtcra:lossScenarioPeril  ?peril ;
            rtcra:incidentSummary    ?summary ;
            rtcra:belongsToOperation ?op .
  ?op rtcra:operationCode ?operation .
  BIND(ext:cosine_object(?peril, "structural collapse ground subsidence") AS ?score)
  FILTER(?score > 0.45)
}
ORDER BY DESC(?score)
LIMIT 8
"""))
print(f"  {len(r)} results\n")
for row in r:
    print(f"  [{row.operation}]  score={float(row.score):.3f}  [{row.peril}]")
    print(f"            {str(row.summary)[:90]}")


# ── Q5: top_k — top 5 risk management sub-elements related to 'auditing' ───
section("Q5 · Top sub-element types closest to 'safety auditing and compliance'")

# cosine_object on the short name strings; score cache means each unique ?name
# is scored only once regardless of how many instances share that name.
r = list(g.query(PREFIX + """
SELECT DISTINCT ?name ?score WHERE {
  ?sub rtcra:riskManagementSubElementName ?name .
  BIND(ext:cosine_object(?name, "safety auditing and compliance verification") AS ?score)
  FILTER(?score > 0.45)
}
ORDER BY DESC(?score)
"""))
print(f"  {len(r)} distinct sub-element types above threshold 0.45\n")
for row in r:
    print(f"  score={float(row.score):.3f}  [{row.name}]")


# ── Q6: Equipment type semantic search ────────────────────────────────────
section("Q6 · Equipment semantically related to 'conveyor and material handling'")

r = list(g.query(PREFIX + """
SELECT ?name ?eqtype ?operation ?score WHERE {
  ?eq rtcra:equipmentName ?name ;
      rtcra:equipmentType ?eqtype ;
      rtcra:belongsToOperation ?op .
  ?op rtcra:operationCode ?operation .
  BIND(ext:cosine_object(?eqtype, "conveyor belt material handling system") AS ?score)
  FILTER(?score > 0.55)
}
ORDER BY DESC(?score)
LIMIT 8
"""))
print(f"  {len(r)} results\n")
for row in r:
    print(f"  [{row.operation}]  score={float(row.score):.3f}  type=[{row.eqtype}]  name={row.name}")


# ── Summary ───────────────────────────────────────────────────────────────
print(f"\n{'═' * 70}")
print(f"  Load: {load_time:.1f}s   Index: {idx_time:.1f}s   Graph: {len(g):,} triples")
print('═' * 70)
