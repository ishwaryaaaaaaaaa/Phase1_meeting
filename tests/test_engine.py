"""Tests for the debate engine. Run:  python -m pytest tests -q

1. Whiteboard reproduction: the exact numbers from the 8 Oct notes come out of the engine.
2. The key identity  m_F + m_E + m_R = S_ALL(B) - S_ALL(A)  on random worlds.
3. Debate winner == full-agent winner for every pair and for the whole tournament.
4. Early stopping with the tight bound NEVER changes a winner (and the loose bounds never do either).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import engine as E  # noqa: E402

# ------------------------------------------------------------------ whiteboard world (3-D)
G_BOARD = np.array([[1.0, -0.6, 0.2],
                    [-0.6, 1.0, -0.05],
                    [0.2, -0.05, 1.0]])
BETA_BOARD = np.array([0.40, 0.35, 0.25])


def board_directions():
    """Unit directions whose Gram matrix is exactly the whiteboard's G (Cholesky factor rows)."""
    return np.linalg.cholesky(G_BOARD)


def test_board_G_is_a_valid_agreement_matrix():
    assert np.all(np.linalg.eigvalsh(G_BOARD) > 0)
    D = board_directions()
    assert np.allclose(np.linalg.norm(D, axis=1), 1.0)
    assert np.allclose(D @ D.T, G_BOARD)


def board_pair_world():
    """Two companies A, B whose margins are exactly the whiteboard's [+0.12, -0.04, +0.07]."""
    D = board_directions()
    target_margins = np.array([0.12, -0.04, 0.07])
    u = target_margins / BETA_BOARD                       # dz . d_k that gives those margins
    dz = np.linalg.solve(D, u)                            # dz such that D @ dz = u
    zA = np.zeros(3)
    return E.World(["A", "B"], np.vstack([zA, zA + dz]), D)


def test_whiteboard_margins_order_and_running_evidence():
    w = board_pair_world()
    res = E.debate_pair(w, BETA_BOARD, 0, 1)
    assert np.allclose(res.margins, [0.12, -0.04, 0.07], atol=1e-12)
    assert res.order == [0, 1, 2]                          # Finance, then ESG (G_FE = -0.6), then Risk
    assert np.allclose(res.running, [0.12, 0.08, 0.15], atol=1e-12)
    assert res.winner == "B" and res.loser == "A"
    E.check_identity(res)
    assert np.isclose(res.full_diff, 0.15)                 # NOTE: the notes' S_ALL(.91) - S_ALL(.81) = 0.10
    #                                                        is a different illustrative example - see EXPLAIN.md


def test_second_speaker_is_strongest_challenger():
    # With the board matrix, after Finance the most negative G is ESG (-0.6) not Risk (+0.2)
    assert E.speaker_order(BETA_BOARD, G_BOARD) == [0, 1, 2]
    # If Risk had the highest weight, ESG (G_RE = -0.05) is more negative than Finance (+0.2)
    assert E.speaker_order(np.array([0.2, 0.2, 0.6]), G_BOARD) == [2, 1, 0]


def board_tournament_world():
    """Four companies with overall full-agent scores A=.81 B=.91 C=.84 D=.76 (the notes)."""
    D = board_directions()
    d_all = E.full_direction(D, BETA_BOARD)
    scores = np.array([0.81, 0.91, 0.84, 0.76])
    Z = np.outer(scores, d_all / (d_all @ d_all))          # then S_ALL(i) = scores[i] exactly
    return E.World(["A", "B", "C", "D"], Z, D), scores


def test_whiteboard_tournament_bracket():
    w, scores = board_tournament_world()
    assert np.allclose(E.full_scores(w, BETA_BOARD), scores)
    t = E.tournament(w, BETA_BOARD, [0, 1, 2, 3])
    assert t.rounds == [["A", "B", "C", "D"], ["C", "D", "B"], ["B", "C"], ["B"]]
    assert [(s.a, s.b, s.winner) for s in t.steps] == [("A", "B", "B"), ("C", "D", "C"), ("B", "C", "B")]
    assert t.winner == "B"
    assert t.messages == 9                                  # 3 debates x 3 agents


# ------------------------------------------------------------------ random worlds
def random_world(rng, n=12, dim=40, corr=0.4):
    base = rng.normal(size=dim)
    D = np.vstack([corr * base + rng.normal(size=dim) for _ in range(3)])
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    Z = rng.normal(size=(n, dim))
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    return E.World([f"c{i}" for i in range(n)], Z, D)


def random_beta(rng):
    b = rng.random(3) + 0.05
    return b / b.sum()


def test_identity_and_pairwise_agreement_with_full_agent():
    rng = np.random.default_rng(0)
    for _ in range(40):
        w, beta = random_world(rng), random_beta(rng)
        s_all = E.full_scores(w, beta)
        for ia in range(len(w.names)):
            for ib in range(ia + 1, len(w.names)):
                res = E.debate_pair(w, beta, ia, ib)
                E.check_identity(res)
                assert (res.winner == w.names[ib]) == (s_all[ib] > s_all[ia])


def test_tournament_winner_is_full_agent_winner_for_any_queue_order():
    rng = np.random.default_rng(1)
    for _ in range(40):
        w, beta = random_world(rng), random_beta(rng)
        cand = list(rng.permutation(len(w.names))[:5])
        t = E.tournament(w, beta, cand)
        assert t.winner == w.names[cand[int(np.argmax(E.full_scores(w, beta)[cand]))]]


def test_early_stopping_never_changes_the_winner():
    rng = np.random.default_rng(2)
    saved = {"loose": 0, "cs": 0, "tight": 0}
    total = 0
    for _ in range(60):
        w, beta = random_world(rng, corr=rng.uniform(0, 1.5)), random_beta(rng)
        for ia in range(len(w.names)):
            for ib in range(ia + 1, len(w.names)):
                full = E.debate_pair(w, beta, ia, ib)
                total += 1
                for kind in saved:
                    r = E.debate_pair_early_stop(w, beta, ia, ib, kind)
                    assert r["winner"] == full.winner, (kind, w.names[ia], w.names[ib])
                    saved[kind] += 3 - r["messages"]
    # tighter bound should save at least as many messages as the looser ones
    assert saved["tight"] >= saved["cs"] >= saved["loose"]


def test_majority_vote_can_disagree_with_full_agent():
    """Two agents mildly prefer A, one agent strongly prefers B and has the largest weight."""
    D = np.eye(3)
    w = E.World(["A", "B"], np.array([[0.2, 0.2, 1.0], [0.0, 0.0, 0.0]]), D)  # (not unit rows; fine for the test)
    beta = np.array([0.1, 0.1, 0.8])
    # dz = zB - zA = [-.2, -.2, -1] -> every agent prefers A here; flip to make a split instead
    w = E.World(["A", "B"], np.array([[0.0, 0.0, 0.0], [0.2, 0.2, -1.0]]), D)
    full = E.debate_pair(w, beta, 0, 1)             # evidence = .1*.2 + .1*.2 + .8*(-1) < 0 -> A wins
    maj = E.decide_majority(w, beta, 0, 1)          # votes: F->B, E->B, R->A => majority says B
    assert full.winner == "A" and maj.winner == "B"
