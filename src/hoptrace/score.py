"""Selection over the expanded candidate pool: ``interleave`` (default;
deterministic merge of seeds and hop rings) and ``submodular`` (experimental;
greedy relevance × novelty over coverage features, kept behind a switch for
measurement — ADR 0008).
"""

from __future__ import annotations

from dataclasses import dataclass

from hoptrace.expand import Candidate


def interleave(pool: dict[int, Candidate], k: int) -> list[Candidate]:
    """Top-k: even slots from ring 0 (seeds), odd slots round-robin over
    rings 1..n; exhausted queues cede their turns."""
    queues: dict[int, list[Candidate]] = {}
    for candidate in pool.values():
        queues.setdefault(candidate.hop, []).append(candidate)
    for queue in queues.values():
        queue.sort(key=lambda c: (-c.score, c.chunk_id))
    seeds = queues.get(0, [])
    hop_rings = [queues[ring] for ring in sorted(queues) if ring >= 1]

    merged: list[Candidate] = []
    si = 0
    positions = [0] * len(hop_rings)
    ring_turn = 0

    def next_hop() -> Candidate | None:
        nonlocal ring_turn
        for _ in range(len(hop_rings)):
            ring = ring_turn % len(hop_rings) if hop_rings else 0
            ring_turn += 1
            if hop_rings and positions[ring] < len(hop_rings[ring]):
                candidate = hop_rings[ring][positions[ring]]
                positions[ring] += 1
                return candidate
        return None

    while len(merged) < k:
        took = False
        if si < len(seeds):
            merged.append(seeds[si])
            si += 1
            took = True
        if len(merged) < k:
            hop_candidate = next_hop()
            if hop_candidate is not None:
                merged.append(hop_candidate)
                took = True
        if not took:
            break
    return merged


#: ("term", query term present in chunk) | ("ent", bridge entity on path)
Feature = tuple[str, str]


@dataclass(frozen=True)
class Selected:
    candidate: Candidate
    novelty: float
    gain: float


def candidate_features(
    candidate: Candidate, chunk_terms: frozenset[str], query_terms: frozenset[str]
) -> frozenset[Feature]:
    """Coverage features: query terms present in the chunk + entities on the hop path.

    Path-entity features are minted by the pool itself, so distinct junk routes
    each earn novelty; query-anchored variants starve true multi-hop targets.
    """
    terms = {("term", t) for t in chunk_terms & query_terms}
    entities = {("ent", e.entity) for e in candidate.path.edges if e.entity}
    return frozenset(terms | entities)


def greedy_select(
    pool: dict[int, Candidate],
    features: dict[int, frozenset[Feature]],
    k: int,
) -> list[Selected]:
    """Greedy marginal-coverage selection; falls back to relevance order
    once every remaining candidate's coverage is exhausted."""
    remaining = sorted(pool.values(), key=lambda c: (-c.score, c.chunk_id))
    covered: set[Feature] = set()
    selected: list[Selected] = []
    while remaining and len(selected) < k:
        best_index = 0
        best_key = (-1.0, -1.0, 0)
        best_novelty = 0.0
        for index, candidate in enumerate(remaining):
            feats = features.get(candidate.chunk_id, frozenset())
            novelty = len(feats - covered) / len(feats) if feats else 0.0
            key = (candidate.score * novelty, candidate.score, -candidate.chunk_id)
            if key > best_key:
                best_key = key
                best_index = index
                best_novelty = novelty
        candidate = remaining.pop(best_index)
        selected.append(Selected(candidate=candidate, novelty=best_novelty, gain=best_key[0]))
        covered |= features.get(candidate.chunk_id, frozenset())
    return selected
