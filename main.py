from rdflib import Literal, URIRef, Namespace
from spicy import SpicyGraph
from spicy.embedder import _get_model

EX = Namespace("https://example.org/mining#")

g = SpicyGraph()

triples = [
    (EX.TailingsPond1,  EX.type, Literal("tailings storage facility")),
    (EX.TailingsPond2,  EX.type, Literal("tailings impoundment")),
    (EX.TailingsPond3,  EX.type, Literal("waste containment pond")),
    (EX.Pit1,           EX.type, Literal("open pit mine")),
    (EX.Pit2,           EX.type, Literal("surface excavation")),
    (EX.Shaft1,         EX.type, Literal("underground mine shaft")),
    (EX.Conveyor1,      EX.type, Literal("ore conveyor belt")),
    (EX.CrushingPlant1, EX.type, Literal("ore crushing facility")),
    (EX.WaterTank1,     EX.type, Literal("water storage tank")),
    (EX.Leach1,         EX.type, Literal("heap leach pad")),
]
for t in triples:
    g.add(t)

print("Building vector index...")
g.enable_vector_search()
model = _get_model()
print(f"Device: {model.device}  |  Index ready.\n")

# ── 1. Cosine threshold filter ────────────────────────────────────────────────
COSINE_QUERY = """
PREFIX ext: <https://spicyrdflib.dev/fn#>
PREFIX ex:  <https://example.org/mining#>

SELECT ?subject ?label ?score WHERE {
  ?subject ex:type ?label .
  BIND(ext:cosine_object(?label, "tailings storage facility") AS ?score)
  FILTER(?score > 0.5)
}
ORDER BY DESC(?score)
"""

print("── cosine_object > 0.5  (query: 'tailings storage facility') ──")
for row in g.query(COSINE_QUERY):
    print(f"  {str(row.subject):<50}  score={float(row.score):.4f}  [{row.label}]")

# ── 2. Top-k filter ───────────────────────────────────────────────────────────
TOPK_QUERY = """
PREFIX ext: <https://spicyrdflib.dev/fn#>
PREFIX ex:  <https://example.org/mining#>

SELECT ?subject ?label WHERE {
  ?subject ex:type ?label .
  FILTER(ext:top_k_object(?label, "mining excavation", 3))
}
"""

print("\n── top_k_object  k=3  (query: 'mining excavation') ──")
for row in g.query(TOPK_QUERY):
    print(f"  {str(row.subject):<50}  [{row.label}]")
