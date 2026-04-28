# SpicyRDFLib — Vector-Augmented SPARQL

SpicyRDFLib extends RDFLib with **semantic similarity search natively inside SPARQL FILTER clauses**.  
Standard SPARQL pattern matching is purely structural and exact. SpicyRDFLib adds cosine-similarity
functions over embedded RDF triple components so you can write queries like:

```sparql
PREFIX ext: <https://spicyrdflib.dev/fn#>

SELECT ?s ?label WHERE {
  ?s ex:type ?label .
  FILTER(ext:cosine_object(?label, "tailings storage facility") > 0.8)
}
```

and get back triples whose object literals are *semantically close* to your query string — not just
exact or substring matches.

---

## Installation

```bash
pip install rdflib
pip install sentence-transformers numpy
```

Clone this repo and install in editable mode:

```bash
git clone https://github.com/imanrahmani78/SpicyRDFLib.git
cd SpicyRDFLib
pip install -e ".[dev]"
```

The first time you call `enable_vector_search()` the embedding model (`all-MiniLM-L6-v2`, ~80 MB)
is downloaded automatically from Hugging Face and cached locally.

---

## Quick Start

```python
from rdflib import Literal, URIRef, Namespace
from spicy import SpicyGraph

EX = Namespace("https://example.org/mining#")

g = SpicyGraph()
g.add((EX.TailingsPond1,  EX.type, Literal("tailings storage facility")))
g.add((EX.TailingsPond2,  EX.type, Literal("tailings impoundment")))
g.add((EX.Pit1,           EX.type, Literal("open pit mine")))
g.add((EX.Shaft1,         EX.type, Literal("underground mine shaft")))

# Build the embedding index (one-time per session)
g.enable_vector_search()

results = g.query("""
    PREFIX ext: <https://spicyrdflib.dev/fn#>
    PREFIX ex:  <https://example.org/mining#>

    SELECT ?s ?label ?score WHERE {
      ?s ex:type ?label .
      BIND(ext:cosine_object(?label, "tailings storage facility") AS ?score)
      FILTER(?score > 0.5)
    }
    ORDER BY DESC(?score)
""")

for row in results:
    print(f"{row.subject}  score={float(row.score):.4f}  [{row.label}]")
```

Output:

```
https://example.org/mining#TailingsPond1  score=1.0000  [tailings storage facility]
https://example.org/mining#TailingsPond2  score=0.6436  [tailings impoundment]
```

---

## How It Works

### 1. Indexing

When you call `enable_vector_search()`, SpicyRDFLib iterates all triples and embeds each unique
**subject**, **predicate**, and **object** component separately using
[`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
— a 22M-parameter model that runs on CPU or Apple MPS without a GPU.

All terms for a given component type are encoded in a **single batched call** to the model, then
stacked into a `numpy` matrix (`N × 384`). Indexing 10,000 triples takes under one second after
the model is loaded.

### 2. Query-time scoring

When a FILTER like `ext:cosine_object(?label, "query") > 0.8` is evaluated, SpicyRDFLib:

1. Embeds `"query"` (cached after the first call)
2. Multiplies the entire object matrix by the query vector in **one `numpy` matmul** — all cosine
   scores for all objects at once
3. Caches the resulting `{term → score}` dict for the duration of the query
4. Every subsequent FILTER row for the same query string is a cheap `dict.get()` lookup

This means the cost of evaluating a cosine FILTER over N triples is dominated by **one matrix
multiply**, not N individual dot products.

### 3. MPS / CUDA acceleration

The model is loaded on the best available device automatically:

| Hardware | Device used |
|---|---|
| Apple Silicon (M-series) | `mps` |
| NVIDIA GPU | `cuda` |
| Everything else | `cpu` |

---

## API Reference

### `SpicyGraph`

A drop-in subclass of `rdflib.Graph`.

```python
from spicy import SpicyGraph
g = SpicyGraph()
```

#### `enable_vector_search() → None`

Build the full embedding index from the current graph contents and register the SPARQL extension
functions. Call this once after loading your data.

```python
g.parse("data.ttl")
g.enable_vector_search()
```

Subsequent calls to `add()` or `parse()` while vector search is enabled will incrementally update
the index for new triples (with a matrix rebuild to keep scoring consistent).

---

## SPARQL Extension Functions

All functions are registered under the namespace `https://spicyrdflib.dev/fn#`.

```sparql
PREFIX ext: <https://spicyrdflib.dev/fn#>
```

### Cosine similarity — `ext:cosine_subject`, `ext:cosine_predicate`, `ext:cosine_object`

```sparql
ext:cosine_object(?term, "query string")  →  xsd:double  (range: –1.0 to 1.0)
```

Returns the cosine similarity between the embedding of `?term` and the embedding of `"query string"`.
Scores close to `1.0` indicate high semantic similarity.

**Usage with BIND:**

```sparql
SELECT ?s ?label ?score WHERE {
  ?s ex:type ?label .
  BIND(ext:cosine_object(?label, "mine waste containment") AS ?score)
  FILTER(?score > 0.6)
}
ORDER BY DESC(?score)
```

**Threshold recommendations:**

| Score range | Interpretation |
|---|---|
| `> 0.9` | Near-identical meaning |
| `0.7 – 0.9` | Strongly related |
| `0.5 – 0.7` | Loosely related |
| `< 0.5` | Unlikely to be relevant |

---

### Top-k membership — `ext:top_k_subject`, `ext:top_k_predicate`, `ext:top_k_object`

```sparql
ext:top_k_object(?term, "query string", k)  →  xsd:boolean
```

Returns `true` if `?term` is among the **k most similar** terms to `"query string"` in its
component index. The top-k set is computed once per `(component, query, k)` combination and cached.

**Usage:**

```sparql
SELECT ?s ?label WHERE {
  ?s ex:type ?label .
  FILTER(ext:top_k_object(?label, "mining excavation", 3))
}
```

Use `top_k_*` when you want a fixed number of results regardless of score distribution, and
`cosine_*` when you want a calibrated threshold.

---

## Performance

| Graph size | Unique terms | Index build | Query (10k rows, all scored) |
|---|---|---|---|
| 100 triples | 220 | 3.3 s* | — |
| 500 triples | 1,020 | 0.26 s | ~50 ms |
| 1,000 triples | 2,020 | 0.35 s | ~40 ms |
| 5,000 triples | 10,020 | 0.61 s | ~190 ms |
| 10,000 triples | 20,020 | 0.78 s | ~400 ms |

\* First call includes ~3 s one-time model load. Subsequent calls on the same process are fast.

Tested on Apple M4 Max with MPS acceleration.

---

## Architecture

```
spicy/
├── __init__.py       # Public API: SpicyGraph, VectorIndex, embed
├── embedder.py       # Lazy model load, per-string cache, batch_embed()
├── index.py          # VectorIndex: build(), add_triple(), get_all_scores(), top_k()
├── functions.py      # SPARQL function registration (cosine_*, top_k_*)
└── graph.py          # SpicyGraph(rdflib.Graph) subclass
```

All code lives under `spicy/` and makes no modifications to RDFLib internals.
Extension functions are registered via RDFLib's existing
[`register_custom_function`](https://rdflib.readthedocs.io/en/latest/) mechanism.

---

## Design Goals and Constraints

- **No external vector database** — everything is in-memory using `numpy`
- **Offline after first model download** — no network calls at query time
- **No RDFLib core modifications** — all new code under `spicy/`
- **SPARQL as the primary interface** — no parallel Python similarity API
- **Python 3.10+**

---

## Roadmap

This is a research proof-of-concept. Planned future work:

- [ ] Persistent index serialisation (avoid re-embedding on restart)
- [ ] ANN index (HNSW via `hnswlib`) for 100k+ triple graphs
- [ ] Full triple embedding as a unit (not just per-component)
- [ ] `ext:similar_triples(?s, ?p, ?o, k)` table-valued function
- [ ] Proposal to the [RDFox team](https://www.oxfordsemantic.tech/) for native production integration
