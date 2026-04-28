# SpicyRDFLib — Claude Code Project Context

## Project Goal

This project is a research fork of [RDFLib](https://github.com/RDFLib/rdflib) to build a proof-of-concept for **vector-augmented SPARQL queries**. The core idea is to embed RDF triple components (subject, predicate, object) using a lightweight on-device embedding model, index those embeddings, and expose them as callable functions within SPARQL FILTER expressions — enabling semantic/fuzzy similarity search natively in SPARQL.

The goal is to validate feasibility in RDFLib before proposing the feature to the RDFox team (Oxford Semantic Technologies) as a production-grade implementation.

---

## Phase 0 — Fork and Clone (Start Here)

### Step 1: Fork RDFLib on GitHub

- Go to https://github.com/RDFLib/rdflib
- Fork it to the authenticated GitHub account
- Rename the fork to **SpicyRDFLib**
- Keep all branches

### Step 2: Clone to Local Machine

```bash
git clone https://github.com/<your-username>/SpicyRDFLib.git
cd SpicyRDFLib
```

### Step 3: Set Upstream Remote

```bash
git remote add upstream https://github.com/RDFLib/rdflib.git
git fetch upstream
```

### Step 4: Create a Working Branch

```bash
git checkout -b feature/vector-sparql
```

### Step 5: Set Up Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install sentence-transformers numpy
```

---

## Architecture Intent

### The Problem

Standard SPARQL pattern matching is purely structural and exact. There is no native way to query for triples whose components are *semantically similar* to a given string. `FILTER(CONTAINS(?s, "mining"))` only does substring matching — not semantic proximity.

### The Proposed Solution

1. **Embed** each unique subject URI, predicate URI, and object literal/URI at graph load time using `all-MiniLM-L6-v2` (via `sentence-transformers`)
2. **Index** those embeddings in an in-memory vector index keyed by component type (subject / predicate / object)
3. **Expose** custom SPARQL extension functions that perform cosine similarity lookup at FILTER evaluation time, e.g.:

```sparql
PREFIX ext: <https://spicyrdflib.dev/fn#>

SELECT ?s ?p ?o WHERE {
  ?s ?p ?o .
  FILTER(ext:cosine_subject(?s, "tailings storage facility") > 0.8)
}
```

---

## Key Files to Understand First

Before making any changes, read and understand these files in the cloned repo:

| File | Why |
|------|-----|
| `rdflib/plugins/sparql/evaluate.py` | Core SPARQL evaluation engine — where FILTER expressions are resolved |
| `rdflib/plugins/sparql/functions.py` | Built-in SPARQL functions — this is where custom functions get registered |
| `rdflib/plugins/sparql/parser.py` | SPARQL parser — relevant if query syntax needs extending |
| `rdflib/graph.py` | Graph object — where triple storage and access live |
| `rdflib/term.py` | URIRef, Literal, BNode definitions — the atomic types we will embed |

---

## Components to Build

### 1. `spicy/embedder.py`
- Load `all-MiniLM-L6-v2` once at startup
- Expose `embed(text: str) -> np.ndarray`
- Cache embeddings by string value to avoid recomputation

### 2. `spicy/index.py`
- Accept a populated `rdflib.Graph`
- Iterate all triples
- Embed subjects, predicates, and objects separately
- Store in three dictionaries: `subject_index`, `predicate_index`, `object_index`
- Each maps `term_string -> embedding_vector`

### 3. `spicy/functions.py`
- Register custom SPARQL extension functions under namespace `https://spicyrdflib.dev/fn#`
- Functions: `cosine_subject`, `cosine_predicate`, `cosine_object`
- Each takes a term and a query string, looks up the term's embedding, computes cosine similarity, returns an `rdflib.term.Literal` float
- Must have access to the active index at evaluation time (pass via closure or a global registry)

### 4. `spicy/graph.py`
- Subclass or wrap `rdflib.Graph`
- Override or hook into `add()` and `parse()` to trigger incremental index updates
- Expose `enable_vector_search()` method that builds the full index and registers the extension functions

---

## Embedding Model

- **Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Why:** 22M parameters, CPU-friendly, ~80MB, strong semantic quality for short text, runs on any hardware without GPU
- **Input:** Raw string form of URIs and literals (strip namespace prefixes for cleaner embeddings where appropriate)
- **Similarity metric:** Cosine similarity

---

## Constraints and Principles

- No external database or vector store — everything in-memory for PoC simplicity
- Must run entirely offline after model download
- Do not break existing RDFLib tests or APIs — all new code lives under `spicy/`
- Python 3.10+
- Mac M4 Max is the primary dev machine (MPS acceleration available if needed later)
- Keep the SPARQL interface as the primary query surface — no parallel Python API for similarity search

---

## Success Criteria for PoC

- A populated RDFGraph can be indexed in under a few seconds for graphs up to ~10k triples
- A SPARQL query using `ext:cosine_subject`, `ext:cosine_predicate`, or `ext:cosine_object` in a FILTER clause returns semantically relevant triples
- The similarity threshold is configurable within the query itself
- No modifications to RDFLib's core parser are required (custom functions registered via the existing extension mechanism)

---

## Out of Scope for PoC

- Persistent embedding storage (no vector DB, no disk cache)
- Full triple embedding as a unit (component-level only for now)
- GPU acceleration
- SPARQL syntax changes (must work within existing FILTER + custom function pattern)
- Any UI or API layer

---

## Next Steps After PoC

Once the PoC is validated, the intent is to present this feature to the **RDFox team at Oxford Semantic Technologies** as a proposal for native integration into RDFox — leveraging their high-performance C++ engine with proper ANN indexing (e.g. HNSW) for production-scale semantic triple search.