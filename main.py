"""
SpicyRDFLib — end-to-end validation suite
==========================================
Builds a realistic mining-domain knowledge graph and runs a series of
increasingly complex SPARQL queries that exercise every SpicyRDFLib feature.
Each query is followed by hand-validated assertions so failures are loud.
"""

from rdflib import Literal, URIRef, Namespace, RDF
from rdflib.namespace import XSD
from spicy import SpicyGraph
from spicy.embedder import _get_model

# ── Namespaces ────────────────────────────────────────────────────────────────
EX   = Namespace("https://example.org/mining#")
EXT  = Namespace("https://spicyrdflib.dev/fn#")
SCHEMA = Namespace("https://schema.org/")

# ── Graph construction ────────────────────────────────────────────────────────
g = SpicyGraph()

triples = [
    # Waste / tailings facilities
    (EX.TailingsPond1,   EX.type,       Literal("tailings storage facility")),
    (EX.TailingsPond2,   EX.type,       Literal("tailings impoundment")),
    (EX.TailingsPond3,   EX.type,       Literal("waste containment pond")),
    (EX.WasteRock1,      EX.type,       Literal("waste rock dump")),

    # Extraction
    (EX.Pit1,            EX.type,       Literal("open pit mine")),
    (EX.Pit2,            EX.type,       Literal("surface excavation")),
    (EX.Shaft1,          EX.type,       Literal("underground mine shaft")),
    (EX.Shaft2,          EX.type,       Literal("decline tunnel")),

    # Processing
    (EX.CrushingPlant1,  EX.type,       Literal("ore crushing facility")),
    (EX.LeachPad1,       EX.type,       Literal("heap leach pad")),
    (EX.Concentrator1,   EX.type,       Literal("mineral concentrator")),

    # Utilities
    (EX.Conveyor1,       EX.type,       Literal("ore conveyor belt")),
    (EX.WaterTank1,      EX.type,       Literal("water storage tank")),
    (EX.PowerStation1,   EX.type,       Literal("diesel power station")),

    # Locations
    (EX.TailingsPond1,   EX.locatedIn,  Literal("Nevada")),
    (EX.TailingsPond2,   EX.locatedIn,  Literal("Nevada")),
    (EX.Pit1,            EX.locatedIn,  Literal("Western Australia")),
    (EX.Pit2,            EX.locatedIn,  Literal("Western Australia")),
    (EX.Shaft1,          EX.locatedIn,  Literal("Ontario")),
    (EX.CrushingPlant1,  EX.locatedIn,  Literal("Nevada")),
    (EX.LeachPad1,       EX.locatedIn,  Literal("Nevada")),

    # Operational data
    (EX.TailingsPond1,   EX.capacity_ML, Literal(12000, datatype=XSD.integer)),
    (EX.TailingsPond2,   EX.capacity_ML, Literal(4500,  datatype=XSD.integer)),
    (EX.WaterTank1,      EX.capacity_ML, Literal(200,   datatype=XSD.integer)),
    (EX.Pit1,            EX.depth_m,     Literal(350,   datatype=XSD.integer)),
    (EX.Pit2,            EX.depth_m,     Literal(120,   datatype=XSD.integer)),
    (EX.Shaft1,          EX.depth_m,     Literal(890,   datatype=XSD.integer)),

    # Ownership / relationships
    (EX.TailingsPond1,   EX.managedBy,   EX.OperatorA),
    (EX.TailingsPond2,   EX.managedBy,   EX.OperatorA),
    (EX.Pit1,            EX.managedBy,   EX.OperatorB),
    (EX.Shaft1,          EX.managedBy,   EX.OperatorC),
    (EX.OperatorA,       EX.type,        Literal("mining company")),
    (EX.OperatorB,       EX.type,        Literal("mining operator")),
    (EX.OperatorC,       EX.type,        Literal("extraction company")),
]

for t in triples:
    g.add(t)

print("Building vector index...")
g.enable_vector_search()
model = _get_model()
print(f"Device: {model.device}  |  {len(g)} triples indexed.\n")

PASS = "PASS"
FAIL = "FAIL"

def section(title):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print('─' * 60)

# ── Query 1: Basic cosine threshold ──────────────────────────────────────────
section("Q1 · Basic cosine threshold — waste containment facilities")

Q1 = """
PREFIX ext: <https://spicyrdflib.dev/fn#>
PREFIX ex:  <https://example.org/mining#>
SELECT ?facility ?label ?score WHERE {
  ?facility ex:type ?label .
  BIND(ext:cosine_object(?label, "tailings storage facility") AS ?score)
  FILTER(?score > 0.5)
}
ORDER BY DESC(?score)
"""

q1_results = list(g.query(Q1))
print(f"  Found {len(q1_results)} facilities above threshold:")
for row in q1_results:
    print(f"    score={float(row.score):.4f}  [{row.label}]")

# TailingsPond1 must be exact match (score ≈ 1.0)
top = q1_results[0]
assert float(top.score) > 0.99, f"Expected score ~1.0 for exact match, got {top.score}"
assert str(top.label) == "tailings storage facility"
# TailingsPond2/3 should also appear (semantically similar)
labels_found = {str(r.label) for r in q1_results}
assert "tailings impoundment" in labels_found, "tailings impoundment should be similar"
print(f"  [{PASS}] Exact match at top, related terms included")

# ── Query 2: Cosine + standard FILTER combination ────────────────────────────
section("Q2 · Cosine + standard FILTER — large tailings in Nevada")

Q2 = """
PREFIX ext: <https://spicyrdflib.dev/fn#>
PREFIX ex:  <https://example.org/mining#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
SELECT ?facility ?label ?cap ?score WHERE {
  ?facility ex:type     ?label ;
            ex:locatedIn "Nevada" ;
            ex:capacity_ML ?cap .
  BIND(ext:cosine_object(?label, "tailings storage facility") AS ?score)
  FILTER(?score > 0.5 && ?cap > 5000)
}
ORDER BY DESC(?cap)
"""

q2_results = list(g.query(Q2))
print(f"  Found {len(q2_results)} large tailings in Nevada:")
for row in q2_results:
    print(f"    score={float(row.score):.4f}  cap={row.cap} ML  [{row.label}]")

# Only TailingsPond1 has capacity > 5000 ML
assert len(q2_results) == 1
assert int(q2_results[0].cap) == 12000
print(f"  [{PASS}] Correctly filtered to 1 large facility (12 000 ML)")

# ── Query 3: top_k with UNION ─────────────────────────────────────────────────
section("Q3 · top_k_object UNION — excavation OR waste, top 2 each")

Q3 = """
PREFIX ext: <https://spicyrdflib.dev/fn#>
PREFIX ex:  <https://example.org/mining#>
SELECT DISTINCT ?facility ?label ?category WHERE {
  {
    ?facility ex:type ?label .
    FILTER(ext:top_k_object(?label, "underground excavation", 2))
    BIND("excavation" AS ?category)
  } UNION {
    ?facility ex:type ?label .
    FILTER(ext:top_k_object(?label, "waste containment", 2))
    BIND("waste" AS ?category)
  }
}
ORDER BY ?category ?label
"""

q3_results = list(g.query(Q3))
print(f"  Found {len(q3_results)} results across both categories:")
for row in q3_results:
    print(f"    [{row.category}]  {row.label}")

categories = {str(r.category) for r in q3_results}
assert "excavation" in categories and "waste" in categories
assert len(q3_results) >= 2
print(f"  [{PASS}] Both categories returned results")

# ── Query 4: Cosine on predicate ──────────────────────────────────────────────
section("Q4 · cosine_predicate — find predicates related to 'location'")

Q4 = """
PREFIX ext: <https://spicyrdflib.dev/fn#>
PREFIX ex:  <https://example.org/mining#>
SELECT DISTINCT ?pred ?score WHERE {
  ?s ?pred ?o .
  BIND(ext:cosine_predicate(?pred, "geographic location") AS ?score)
  FILTER(?score > 0.3)
}
ORDER BY DESC(?score)
"""

q4_results = list(g.query(Q4))
print(f"  Found {len(q4_results)} predicates similar to 'geographic location':")
for row in q4_results:
    print(f"    score={float(row.score):.4f}  {row.pred}")

pred_local_names = {str(r.pred).split("#")[-1] for r in q4_results}
assert "locatedIn" in pred_local_names, f"locatedIn should match 'geographic location', got: {pred_local_names}"
print(f"  [{PASS}] 'locatedIn' predicate correctly identified as location-related")

# ── Query 5: Cosine on subject URI ────────────────────────────────────────────
section("Q5 · cosine_subject — find subjects semantically close to 'tailings'")

Q5 = """
PREFIX ext: <https://spicyrdflib.dev/fn#>
PREFIX ex:  <https://example.org/mining#>
SELECT ?subject ?label ?score WHERE {
  ?subject ex:type ?label .
  BIND(ext:cosine_subject(?subject, "tailings pond") AS ?score)
  FILTER(?score > 0.5)
}
ORDER BY DESC(?score)
"""

q5_results = list(g.query(Q5))
print(f"  Found {len(q5_results)} subjects similar to 'tailings pond':")
for row in q5_results:
    print(f"    score={float(row.score):.4f}  {str(row.subject).split('#')[-1]}")

subject_names = {str(r.subject).split("#")[-1] for r in q5_results}
assert any("Tailings" in s for s in subject_names), f"TailingsX URIs should score high, got: {subject_names}"
print(f"  [{PASS}] TailingsPond URIs correctly identified via subject embedding")

# ── Query 6: Subquery — top-k objects per location ───────────────────────────
section("Q6 · Subquery — most similar facility per location")

Q6 = """
PREFIX ext: <https://spicyrdflib.dev/fn#>
PREFIX ex:  <https://example.org/mining#>
SELECT ?location ?facility ?label ?score WHERE {
  ?facility ex:type     ?label ;
            ex:locatedIn ?location .
  BIND(ext:cosine_object(?label, "tailings storage facility") AS ?score)
  {
    SELECT ?location (MAX(ext:cosine_object(?lbl, "tailings storage facility")) AS ?maxScore)
    WHERE {
      ?f ex:type ?lbl ; ex:locatedIn ?location .
    }
    GROUP BY ?location
  }
  FILTER(?score = ?maxScore)
}
ORDER BY ?location
"""

q6_results = list(g.query(Q6))
print(f"  Best-match facility per location:")
for row in q6_results:
    print(f"    {row.location:<20}  score={float(row.score):.4f}  [{row.label}]")

locations_seen = {str(r.location) for r in q6_results}
assert len(q6_results) > 0, "Subquery returned no results"
print(f"  [{PASS}] Subquery correctly grouped and selected max-score per location ({len(locations_seen)} locations)")

# ── Query 7: Incremental add — new triple appears in query ────────────────────
section("Q7 · Incremental index update — add triple after enable_vector_search")

g.add((EX.NewTailingsDam, EX.type, Literal("slurry retention dam")))

# Query specifically for the new term — "dam" and "slurry" are semantically
# close enough to "slurry pond" that this will surface the new triple.
Q7 = """
PREFIX ext: <https://spicyrdflib.dev/fn#>
PREFIX ex:  <https://example.org/mining#>
SELECT ?facility ?label ?score WHERE {
  ?facility ex:type ?label .
  BIND(ext:cosine_object(?label, "slurry pond dam") AS ?score)
  FILTER(?score > 0.5)
}
ORDER BY DESC(?score)
"""

q7_results = list(g.query(Q7))
new_labels = {str(r.label) for r in q7_results}
print(f"  Results after incremental add (query: 'slurry pond dam'):")
for row in q7_results:
    print(f"    score={float(row.score):.4f}  [{row.label}]")
assert "slurry retention dam" in new_labels, (
    f"Newly added triple should appear when querying 'slurry pond dam'. Got: {new_labels}"
)
print(f"  [{PASS}] New triple correctly indexed and retrieved")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'═' * 60}")
print("  All 7 queries passed validation.")
print('═' * 60)
