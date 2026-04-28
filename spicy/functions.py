from __future__ import annotations
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
# FILTER(ext:cosine_subject(?s, "query") > 0.8)
#
# On the first call for a given query string, scores every term in the
# component with one matrix multiply, then caches the result. Every
# subsequent FILTER row for the same query is a cheap dict lookup.
# ------------------------------------------------------------------

def _make_cosine_fn(component: str):
    def fn(e, ctx):
        _require_index()
        args = e.expr
        if len(args) != 2:
            raise SPARQLError(f"cosine_{component}: expected 2 arguments (term, query_string), got {len(args)}")
        term = args[0]
        query = str(args[1])
        scores = _active_index.get_all_scores(component, query)
        return Literal(scores.get(str(term), 0.0), datatype=XSD.double)

    fn.__name__ = f"cosine_{component}"
    return fn


# ------------------------------------------------------------------
# ext:top_k_subject / top_k_predicate / top_k_object
#
# FILTER(ext:top_k_subject(?s, "query", 5))
#
# Returns true if ?s is among the top-k most similar subjects.
# The top-k set is computed once per (component, query, k) and cached.
# ------------------------------------------------------------------

_top_k_cache: dict[tuple[str, str, int], set[str]] = {}


def _make_top_k_fn(component: str):
    def fn(e, ctx):
        _require_index()
        args = e.expr
        if len(args) != 3:
            raise SPARQLError(f"top_k_{component}: expected 3 arguments (term, query, k), got {len(args)}")
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
