"""Tests for the three triad counting identities and the motif role-name order.

In a simple directed acyclic graph, every two-step path A->B->C is either a
chain (021C) or part of a feedforward loop (030T); every pair of edges into one
node is either a fan-in (021U) or part of a loop; and every pair of edges out
of one node is either a fan-out (021D) or part of a loop. Attribution graphs
are DAGs because every edge goes from an earlier layer to a later one.
"""

from pathlib import Path

import igraph as ig
import pytest

from src.graph_loader import load_attribution_graph
from src.motif_census import triad_identity_gaps

DALLAS_PATH = Path(__file__).resolve().parent.parent / "data" / "examples" / "capital-state-dallas.json"


def _tiny_forward_map() -> ig.Graph:
    """A->B, B->C, A->C, D->E. Nodes 0..4 are A..E. One loop, nothing else."""
    return ig.Graph(n=5, edges=[(0, 1), (1, 2), (0, 2), (3, 4)], directed=True)


def _tiny_map_with_backward_arrow() -> ig.Graph:
    """The tiny map plus C->A, which makes A and C a two-way pair."""
    g = _tiny_forward_map()
    g.add_edge(2, 0)
    return g


@pytest.fixture(scope="module")
def dallas_graph() -> ig.Graph:
    return load_attribution_graph(DALLAS_PATH)


def test_identities_hold_on_tiny_forward_map():
    """A->B, B->C, A->C, D->E: all three counting rules hold and every gap is 0."""
    result = triad_identity_gaps(_tiny_forward_map())

    # Hand count: the only loop is A,B,C; no chain, fan-in or fan-out triads.
    assert (result.chains, result.loops, result.fan_ins, result.fan_outs) == (0, 1, 0, 0)
    # Hand count: one two-step path A->B->C, one pair into C, one pair out of A.
    assert (result.two_step_paths, result.pairs_in, result.pairs_out) == (1, 1, 1)
    assert result.gaps == (0, 0, 0)


def test_identities_break_on_map_with_backward_arrow():
    """Same tiny map plus C->A: at least one gap is not 0, so the check can catch a broken rule."""
    result = triad_identity_gaps(_tiny_map_with_backward_arrow())

    # Hand count: A->B->C, B->C->A, A->C->A, C->A->B, C->A->C.
    assert result.two_step_paths == 5
    # The A,B,C triad now has a two-way pair, so it is none of the four shapes.
    assert (result.chains, result.loops) == (0, 0)
    assert result.gap_chain == -5
    assert result.gaps != (0, 0, 0)


def test_identities_hold_on_dallas_with_published_counts(dallas_graph):
    """Bundled Dallas graph: chains 10,150 + loops 16,125 = 26,275 paths; fan-ins 12,884 + loops = 29,009 pairs in; fan-outs 6,063 + loops = 22,188 pairs out."""
    result = triad_identity_gaps(dallas_graph)

    assert (result.chains, result.loops, result.fan_ins, result.fan_outs) == (10_150, 16_125, 12_884, 6_063)
    assert (result.two_step_paths, result.pairs_in, result.pairs_out) == (26_275, 29_009, 22_188)
    assert result.gaps == (0, 0, 0)
