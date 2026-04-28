"""
SpicyRDFLib — Combined CRA + ToC real-world query suite
========================================================
Loads all four Rio Tinto ontology files into one SpicyGraph and runs
semantically-rich SPARQL queries that span both datasets.

CRA (rtcra:)  — structured risk data: scenarios, recommendations,
                equipment, risk-management systems.  Many properties
                carry long paragraph-level text that benefits from
                chunk-level embedding:
                  incidentDescription, hardControls, softControls,
                  inherentHazards, biSummary, recoveryRepair,
                  recommendationDetailText, siteResponseOrComments,
                  riskManagementSubElementInformation

ToC (rttoc:)  — document corpus: per-site CRA report table-of-contents
                with shortSummary and longSummary prose (300-2000 chars).
                Cross-graph join key: rttoc:site == rtcra:operationCode

Run once to build the LanceDB index (~60-90s for combined corpus).
Subsequent runs load from cache in ~3-5s.
"""

import os, sys, time, json
sys.path.insert(0, ".")

from spicy import SpicyGraph
from spicy.embedder import _get_model

CRA_SCHEMA    = "Ontology/RioTintoCRA_NonInsV4.ttl"
CRA_INSTANCES = "Ontology/RioTintoCRA_InsV4.1.ttl"
TOC_SCHEMA    = "Ontology/RioTintoToC_NonInsV4.ttl"
TOC_INSTANCES = "Ontology/RioTintoToC_InsV4.ttl"
LANCE_PATH    = ".spicy_cache/combined_index"

PREFIX = """
PREFIX rtcra: <http://riotinto.cra.analytical.ontology#>
PREFIX rttoc: <http://riotinto.toc.ontology#>
PREFIX rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs:  <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd:   <http://www.w3.org/2001/XMLSchema#>
PREFIX ext:   <https://spicyrdflib.dev/fn#>
"""

def section(title):
    print(f"\n{'─' * 72}")
    print(f"  {title}")
    print('─' * 72)

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading CRA + ToC ontologies and knowledge graphs...")
t0 = time.perf_counter()
g = SpicyGraph()
g.parse(CRA_SCHEMA,    format="turtle")
g.parse(CRA_INSTANCES, format="turtle")
g.parse(TOC_SCHEMA,    format="turtle")
g.parse(TOC_INSTANCES, format="turtle")
load_time = time.perf_counter() - t0
print(f"  {len(g):,} triples loaded in {load_time:.1f}s")

print("\nBuilding / loading combined vector index (LanceDB)...")
os.makedirs(os.path.dirname(LANCE_PATH), exist_ok=True)
t0 = time.perf_counter()
g.enable_vector_search(index_path=LANCE_PATH)
idx_time = time.perf_counter() - t0
model = _get_model()
print(f"  Device: {model.device}  |  Index ready in {idx_time:.1f}s")

# ── Q1: CRA incidentDescription — semantic search on long scenario text ────────
section("Q1 · CRA incidentDescription — 'conveyor fire spread to electrical equipment'")

r = list(g.query(PREFIX + """
SELECT ?scenarioID ?site ?year ?totalLoss ?score WHERE {
  ?scenario rdf:type                      rtcra:CriticalRiskScenario ;
            rtcra:criticalRiskScenarioID  ?scenarioID ;
            rtcra:incidentDescription     ?desc ;
            rtcra:totalLossUSDM           ?totalLoss ;
            rtcra:belongsToReportYear     ?year ;
            rtcra:belongsToOperation      ?op .
  ?op rtcra:operationCode ?site .
  BIND(ext:cosine_object(?desc, "conveyor fire spread to electrical equipment switchroom") AS ?score)
  FILTER(?score > 0.45)
}
ORDER BY DESC(?score)
LIMIT 6
"""))
print(f"  {len(r)} results\n")
for row in r:
    print(f"  score={float(row.score):.3f}  [{row.site}]  {row.scenarioID}  "
          f"loss=${float(row.totalLoss):.1f}M  yr={row.year}")


# ── Q2: CRA hardControls — find scenarios with fire-detection controls ─────────
section("Q2 · CRA hardControls — scenarios with 'automatic fire detection and suppression'")

r = list(g.query(PREFIX + """
SELECT DISTINCT ?scenarioID ?site ?peril ?score WHERE {
  ?scenario rdf:type                      rtcra:CriticalRiskScenario ;
            rtcra:criticalRiskScenarioID  ?scenarioID ;
            rtcra:lossScenarioPeril       ?peril ;
            rtcra:hardControls            ?controls ;
            rtcra:belongsToOperation      ?op .
  ?op rtcra:operationCode ?site .
  BIND(ext:cosine_object(?controls, "automatic fire detection suppression system installed") AS ?score)
  FILTER(?score > 0.50)
}
ORDER BY DESC(?score)
LIMIT 8
"""))
print(f"  {len(r)} results\n")
for row in r:
    print(f"  score={float(row.score):.3f}  [{row.site}]  [{row.peril}]  {row.scenarioID}")


# ── Q3: CRA recommendationDetailText — detailed rec text semantic search ───────
section("Q3 · CRA recommendationDetailText — 'install fixed water spray suppression system'")

r = list(g.query(PREFIX + """
SELECT ?recID ?site ?priority ?status ?cost ?score WHERE {
  ?rec rdf:type                              rtcra:Recommendation ;
       rtcra:recommendationID               ?recID ;
       rtcra:recommendationDetailText       ?detail ;
       rtcra:recommendationPriority         ?priority ;
       rtcra:recommendationStatus           ?status ;
       rtcra:belongsToOperation             ?op .
  ?op rtcra:operationCode ?site .
  OPTIONAL { ?rec rtcra:recommendationEstimatedCostUSDM ?cost }
  BIND(ext:cosine_object(?detail, "install fixed water spray fire suppression conveyor") AS ?score)
  FILTER(?score > 0.50)
}
ORDER BY DESC(?score)
LIMIT 6
"""))
print(f"  {len(r)} results\n")
for row in r:
    cost_str = f"${float(row.cost):.2f}M" if row.cost else "n/a"
    print(f"  score={float(row.score):.3f}  [{row.site}]  {row.recID}  "
          f"priority={row.priority}  status={row.status}  cost={cost_str}")


# ── Q4: CRA riskManagementSubElementInformation — inspection & testing ─────────
section("Q4 · CRA subElementInformation — 'inspection testing maintenance schedule'")

r = list(g.query(PREFIX + """
SELECT DISTINCT ?name ?site ?score WHERE {
  ?sub rdf:type                                rtcra:RiskManagementSubElement ;
       rtcra:riskManagementSubElementName      ?name ;
       rtcra:riskManagementSubElementInformation ?info ;
       rtcra:belongsToOperation                ?op .
  ?op rtcra:operationCode ?site .
  BIND(ext:cosine_object(?info, "inspection testing maintenance schedule frequency") AS ?score)
  FILTER(?score > 0.50)
}
ORDER BY DESC(?score)
LIMIT 8
"""))
print(f"  {len(r)} results\n")
for row in r:
    print(f"  score={float(row.score):.3f}  [{row.site}]  [{row.name}]")


# ── Q5: ToC longSummary — deep document section search with chunking ───────────
section("Q5 · ToC longSummary — 'SAG Mill or concentrator catastrophic failure financial impact'")

r = list(g.query(PREFIX + """
SELECT DISTINCT ?site ?reportYear ?reportType ?sectionTitle ?score WHERE {
  ?section rttoc:sectionTitle      ?sectionTitle ;
           rttoc:longSummary       ?summary ;
           rttoc:belongsToDocument ?doc .
  ?doc rttoc:site       ?site ;
       rttoc:reportYear  ?reportYear ;
       rttoc:reportType  ?reportType .
  BIND(ext:cosine_object(?summary, "SAG Mill concentrator failure catastrophic financial loss") AS ?score)
  FILTER(?score > 0.45)
}
ORDER BY DESC(?score)
LIMIT 8
"""))
print(f"  {len(r)} results\n")
for row in r:
    print(f"  score={float(row.score):.3f}  [{row.site}]  {row.reportYear}/{row.reportType}  "
          f"§ {str(row.sectionTitle)[:55]}")


# ── Q6: ToC shortSummary — fire protection across all sites and years ──────────
section("Q6 · ToC shortSummary — 'fire protection and asset protection systems' across all sites")

r = list(g.query(PREFIX + """
SELECT DISTINCT ?site ?reportYear ?sectionTitle ?score WHERE {
  ?section rttoc:sectionTitle      ?sectionTitle ;
           rttoc:shortSummary      ?summary ;
           rttoc:belongsToDocument ?doc .
  ?doc rttoc:site       ?site ;
       rttoc:reportYear  ?reportYear .
  BIND(ext:cosine_object(?summary, "fire protection asset protection system performance") AS ?score)
  FILTER(?score > 0.50)
}
ORDER BY DESC(?score)
LIMIT 8
"""))
print(f"  {len(r)} results\n")
for row in r:
    print(f"  score={float(row.score):.3f}  [{row.site}]  {row.reportYear}"
          f"  § {str(row.sectionTitle)[:55]}")


# ── Q7: Cross-graph — ToC structural sections at CRA high-loss sites ───────────
# Two-step: first collect the qualifying site codes in Python (fast CRA query),
# then inject them as a FILTER IN for the ToC semantic search.
# RDFLib has no query optimizer — FILTER EXISTS iterates all candidates per row.
section("Q7 · Cross-graph — ToC structural/collapse sections at CRA high-loss sites (>$200M)")

_sites_q7 = list(g.query(PREFIX + """
SELECT DISTINCT ?site WHERE {
  ?op rtcra:operationCode ?site .
  ?scenario rtcra:lossScenarioPeril  ?peril ;
            rtcra:totalLossUSDM      ?loss ;
            rtcra:belongsToOperation ?op .
  FILTER(?loss > 200 && CONTAINS(str(?peril), "Collapse"))
}
"""))
_collapse_sites = [str(row.site) for row in _sites_q7]
print(f"  Sites with Collapse scenarios >$200M: {_collapse_sites}\n")

if _collapse_sites:
    _in_clause = ", ".join(f'"{s}"' for s in _collapse_sites)
    r = list(g.query(PREFIX + f"""
    SELECT ?site ?sectionTitle ?docYear ?secScore WHERE {{
      ?section rttoc:sectionTitle      ?sectionTitle ;
               rttoc:longSummary       ?summary ;
               rttoc:belongsToDocument ?doc .
      ?doc rttoc:site       ?site ;
           rttoc:reportYear  ?docYear .
      FILTER(?site IN ({_in_clause}))
      BIND(ext:cosine_object(?summary, "structural failure collapse slope geotechnical") AS ?secScore)
      FILTER(?secScore > 0.44)
    }}
    ORDER BY DESC(?secScore)
    LIMIT 8
    """))
    print(f"  {len(r)} matching ToC sections\n")
    for row in r:
        print(f"  [{row.site}]  {row.docYear}  secScore={float(row.secScore):.3f}"
              f"  § {str(row.sectionTitle)[:60]}")
else:
    print("  No qualifying sites found.")


# ── Q8: Cross-graph — ToC natural-hazard sections at CRA cyclone-exposed sites ─
# Same two-step approach: collect cyclone-exposed sites from CRA, inject as FILTER IN.
section("Q8 · Cross-graph semantic Template 30 — cyclone-exposed sites → ToC hazard sections")

_sites_q8 = list(g.query(PREFIX + """
SELECT DISTINCT ?site WHERE {
  ?op rtcra:operationCode ?site .
  ?report rtcra:belongsToOperation ?op ;
          rtcra:nathanData         ?ndata .
  FILTER(CONTAINS(LCASE(str(?ndata)), "cyclone"))
}
"""))
_cyclone_sites = [str(row.site) for row in _sites_q8]
print(f"  Cyclone-exposed sites from CRA Nathan data: {_cyclone_sites}\n")

if _cyclone_sites:
    _in_clause = ", ".join(f'"{s}"' for s in _cyclone_sites)
    r = list(g.query(PREFIX + f"""
    SELECT ?site ?sectionTitle ?docYear ?secScore WHERE {{
      ?section rttoc:sectionTitle      ?sectionTitle ;
               rttoc:longSummary       ?summary ;
               rttoc:belongsToDocument ?doc .
      ?doc rttoc:site       ?site ;
           rttoc:reportYear  ?docYear .
      FILTER(?site IN ({_in_clause}))
      BIND(ext:cosine_object(?summary, "cyclone wind storm surge natural hazard exposure") AS ?secScore)
      FILTER(?secScore > 0.42)
    }}
    ORDER BY DESC(?secScore)
    LIMIT 8
    """))
    print(f"  {len(r)} matching ToC sections\n")
    for row in r:
        print(f"  [{row.site}]  {row.docYear}  secScore={float(row.secScore):.3f}"
              f"  § {str(row.sectionTitle)[:60]}")
else:
    print("  No cyclone-exposed sites found.")


# ── Q9: cosine_any — scoped to CRA scenarios (avoids full 382k-triple scan) ────
# cosine_any is most useful when you don't know which component carries the signal.
# Here: subject=scenario URI, predicate=property name, object=literal text.
section("Q9 · cosine_any — CRA scenario triples related to 'tailings dam failure geotechnical'")

r = list(g.query(PREFIX + """
SELECT ?s ?p ?o ?score WHERE {
  ?s rdf:type rtcra:CriticalRiskScenario .
  ?s ?p ?o .
  BIND(ext:cosine_any(?s, ?p, ?o, "tailings dam failure geotechnical slope stability") AS ?score)
  FILTER(?score > 0.65)
}
ORDER BY DESC(?score)
LIMIT 8
"""))
print(f"  {len(r)} results\n")
for row in r:
    p_local = str(row.p).split("#")[-1]
    s_local = str(row.s).split("/")[-1][:40]
    o_str   = str(row.o)[:70]
    print(f"  score={float(row.score):.3f}  [{s_local}] --{p_local}--> [{o_str}]")


# ── Q10: explain — breakdown for cross-graph match ────────────────────────────
section("Q10 · explain — per-component breakdown: ToC section vs 'electrical fire switchroom'")

r = list(g.query(PREFIX + """
SELECT ?site ?sectionTitle ?details WHERE {
  ?section rttoc:sectionTitle      ?sectionTitle ;
           rttoc:longSummary       ?summary ;
           rttoc:belongsToDocument ?doc .
  ?doc rttoc:site ?site .
  BIND(ext:cosine_object(?summary, "electrical switchroom substation fire loss") AS ?score)
  BIND(ext:explain(?section, rttoc:longSummary, ?summary,
       "electrical switchroom substation fire loss") AS ?details)
  FILTER(?score > 0.45)
}
ORDER BY DESC(?score)
LIMIT 5
"""))
print(f"  {len(r)} results\n")
for row in r:
    d = json.loads(str(row.details))
    print(f"  [{row.site}]  § {str(row.sectionTitle)[:50]}")
    print(f"    best={d['best']:<9}  score={d['score']:.3f}"
          f"  sub={d['subject']:.3f}  pred={d['predicate']:.3f}  obj={d['object']:.3f}")


# ── Q11: Semantic upgrade of Template 31 — free-text incidentSummary ──────────
# The peril field is a controlled vocabulary ("Fire-Explosion") so CONTAINS = semantic
# there. The real advantage shows on free-text fields like incidentSummary where
# semantic finds related incidents regardless of exact wording.
section("Q11 · Semantic vs CONTAINS on incidentSummary — 'fire' keyword vs cosine")

r_semantic = list(g.query(PREFIX + """
SELECT DISTINCT ?scenarioID ?site ?peril ?totalLoss ?score WHERE {
  ?scenario rdf:type                     rtcra:CriticalRiskScenario ;
            rtcra:criticalRiskScenarioID ?scenarioID ;
            rtcra:lossScenarioPeril      ?peril ;
            rtcra:incidentSummary        ?summary ;
            rtcra:totalLossUSDM          ?totalLoss ;
            rtcra:belongsToOperation     ?op .
  ?op rtcra:operationCode ?site .
  BIND(ext:cosine_object(?summary, "fire ignition explosion combustion electrical") AS ?score)
  FILTER(?score > 0.55)
}
ORDER BY DESC(?score) LIMIT 12
"""))

r_contains = list(g.query(PREFIX + """
SELECT DISTINCT ?scenarioID ?site ?peril ?totalLoss WHERE {
  ?scenario rdf:type                     rtcra:CriticalRiskScenario ;
            rtcra:criticalRiskScenarioID ?scenarioID ;
            rtcra:lossScenarioPeril      ?peril ;
            rtcra:incidentSummary        ?summary ;
            rtcra:totalLossUSDM          ?totalLoss ;
            rtcra:belongsToOperation     ?op .
  ?op rtcra:operationCode ?site .
  FILTER(CONTAINS(LCASE(str(?summary)), "fire"))
}
ORDER BY DESC(?totalLoss) LIMIT 12
"""))

sem_ids  = {str(r.scenarioID) for r in r_semantic}
cont_ids = {str(r.scenarioID) for r in r_contains}
only_semantic = sem_ids - cont_ids
only_contains = cont_ids - sem_ids

print(f"  Semantic cosine > 0.55 on incidentSummary: {len(sem_ids)} unique scenarios")
print(f"  CONTAINS('fire') on incidentSummary:        {len(cont_ids)} unique scenarios")

if only_semantic:
    print(f"\n  Only found by SEMANTIC (missed by CONTAINS — fire-related but no 'fire' word):")
    sem_lookup = {str(r.scenarioID): r for r in r_semantic}
    for sid in sorted(only_semantic):
        row = sem_lookup[sid]
        print(f"    [{row.site}]  {sid}  [{row.peril}]  "
              f"score={float(row.score):.3f}  loss=${float(row.totalLoss):.0f}M")
        print(f"      → {str(g.value(subject=None, predicate=None, object=None)) or ''}")

if only_contains:
    print(f"\n  Only found by CONTAINS (too generic for semantic threshold 0.55):")
    for sid in sorted(only_contains)[:5]:
        row = next(r for r in r_contains if str(r.scenarioID) == sid)
        print(f"    [{row.site}]  {sid}  [{row.peril}]  loss=${float(row.totalLoss):.0f}M")


# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n{'═' * 72}")
print(f"  Load: {load_time:.1f}s   Index: {idx_time:.1f}s   "
      f"Graph: {len(g):,} triples")
print('═' * 72)
