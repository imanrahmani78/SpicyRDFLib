from __future__ import annotations
import json
from typing import TYPE_CHECKING
from rdflib import URIRef, Literal
from rdflib.namespace import XSD
from rdflib.plugins.sparql.operators import register_custom_function
from rdflib.plugins.sparql.sparql import SPARQLError

if TYPE_CHECKING:
    from .index import VectorIndex

EXT_NS = "https://spicyrdflib.dev/fn#"

_active_index: VectorIndex | None = None


def set_active_index(index) -> None:
    global _active_index
    _active_index = index


def _require_index():
    if _active_index is None:
        raise SPARQLError("Vector index not initialized — call enable_vector_search() first.")


# ------------------------------------------------------------------
# ext:cosine_subject / cosine_predicate / cosine_object
#
# BIND(ext:cosine_object(?o, "query string") AS ?score)
#
# Scores every unique term in the component with one matmul on the first
# call for a given query string, then caches the result. Every subsequent
# FILTER/BIND row for the same query is a cheap dict lookup.
# ------------------------------------------------------------------

def _make_cosine_fn(component: str):
    def fn(e, ctx):
        _require_index()
        args = e.expr
        if len(args) != 2:
            raise SPARQLError(
                f"cosine_{component}: expected 2 arguments (term, query_string), got {len(args)}"
            )
        term = args[0]
        query = str(args[1])
        scores = _active_index.get_all_scores(component, query)
        return Literal(scores.get(str(term), 0.0), datatype=XSD.double)

    fn.__name__ = f"cosine_{component}"
    return fn


# ------------------------------------------------------------------
# ext:top_k_subject / top_k_predicate / top_k_object
#
# FILTER(ext:top_k_object(?o, "query", 5))
#
# Returns true if ?o is among the top-k most similar objects.
# The top-k set is computed once per (component, query, k) and cached.
# ------------------------------------------------------------------

_top_k_cache: dict[tuple[str, str, int], set[str]] = {}


def _make_top_k_fn(component: str):
    def fn(e, ctx):
        _require_index()
        args = e.expr
        if len(args) != 3:
            raise SPARQLError(
                f"top_k_{component}: expected 3 arguments (term, query, k), got {len(args)}"
            )
        term = args[0]
        query = str(args[1])
        try:
            k = int(args[2])
        except (ValueError, TypeError):
            raise SPARQLError(f"top_k_{component}: k must be an integer, got {args[2]!r}")

        cache_key = (component, query, k)
        if cache_key not in _top_k_cache:
            _top_k_cache[cache_key] = {key for key, _ in _active_index.top_k(component, query, k)}
        return Literal(str(term) in _top_k_cache[cache_key], datatype=XSD.boolean)

    fn.__name__ = f"top_k_{component}"
    return fn


# ------------------------------------------------------------------
# ext:cosine_any(?s, ?p, ?o, "query")
#
# Returns max(subject_score, predicate_score, object_score) for the
# current triple — useful when you don't know which component carries
# the relevant text.
# ------------------------------------------------------------------

def _cosine_any_fn(e, ctx):
    _require_index()
    args = e.expr
    if len(args) != 4:
        raise SPARQLError(
            f"cosine_any: expected 4 arguments (?s, ?p, ?o, query_string), got {len(args)}"
        )
    s_term, p_term, o_term = args[0], args[1], args[2]
    query = str(args[3])
    s_score = _active_index.get_all_scores("subject", query).get(str(s_term), 0.0)
    p_score = _active_index.get_all_scores("predicate", query).get(str(p_term), 0.0)
    o_score = _active_index.get_all_scores("object", query).get(str(o_term), 0.0)
    return Literal(max(s_score, p_score, o_score), datatype=XSD.double)


# ------------------------------------------------------------------
# ext:explain(?s, ?p, ?o, "query")
#
# Returns a JSON string with per-component scores + which component
# matched best:
#   {"subject": 0.45, "predicate": 0.12, "object": 0.78,
#    "best": "object", "score": 0.78}
# ------------------------------------------------------------------

def _explain_fn(e, ctx):
    _require_index()
    args = e.expr
    if len(args) != 4:
        raise SPARQLError(
            f"explain: expected 4 arguments (?s, ?p, ?o, query_string), got {len(args)}"
        )
    s_term, p_term, o_term = args[0], args[1], args[2]
    query = str(args[3])
    s_score = float(_active_index.get_all_scores("subject", query).get(str(s_term), 0.0))
    p_score = float(_active_index.get_all_scores("predicate", query).get(str(p_term), 0.0))
    o_score = float(_active_index.get_all_scores("object", query).get(str(o_term), 0.0))
    best_name, best_score = max(
        [("subject", s_score), ("predicate", p_score), ("object", o_score)],
        key=lambda x: x[1],
    )
    result = {
        "subject": round(s_score, 4),
        "predicate": round(p_score, 4),
        "object": round(o_score, 4),
        "best": best_name,
        "score": round(best_score, 4),
    }
    return Literal(json.dumps(result), datatype=XSD.string)


# ------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------

def register_functions() -> None:
    _top_k_cache.clear()
    for component in ("subject", "predicate", "object"):
        register_custom_function(
            URIRef(EXT_NS + f"cosine_{component}"),
            _make_cosine_fn(component),
            raw=True,
            override=True,
        )
        register_custom_function(
            URIRef(EXT_NS + f"top_k_{component}"),
            _make_top_k_fn(component),
            raw=True,
            override=True,
        )
    register_custom_function(
        URIRef(EXT_NS + "cosine_any"),
        _cosine_any_fn,
        raw=True,
        override=True,
    )
    register_custom_function(
        URIRef(EXT_NS + "explain"),
        _explain_fn,
        raw=True,
        override=True,
    )
