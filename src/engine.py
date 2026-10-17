"""The debate engine (Phase 7) - the logic from the whiteboard, step by step.

Objects
  z_i      embedding of company i (a point in space)
  d_k      direction of perspective k  (F=Finance, E=ESG, R=Risk), unit length
  beta_k   how much the INVESTOR cares about perspective k (>=0, sums to 1)
  G        agent agreement matrix, G_kl = cos(d_k, d_l)   (who tends to challenge whom)

Full agent    d_ALL = sum_k beta_k d_k ;  S_ALL(i) = z_i . d_ALL      (sees everything)
Partial agent k only knows beta_k * d_k.                              (sees one slice)

Debate on a pair A vs B
  dz = z_B - z_A
  margin of agent k  m_k = beta_k * (dz . d_k)      (+ favours B, - favours A)
  speaker order  : 1) highest beta speaks first
                   2) then the strongest CHALLENGER of the last speaker (most negative G with them)
                   3) then the remaining agent(s) the same way          <- inferred from the notes
  running evidence = sum of the margins heard so far; the sign of the final value picks the winner.

Key identity (why the debate agrees with the full agent):
  S_ALL(B) - S_ALL(A) = dz . d_ALL = sum_k beta_k dz . d_k = m_F + m_E + m_R
"""
from dataclasses import dataclass, field

import numpy as np

AGENTS = ["F", "E", "R"]
AGENT_NAME = {"F": "Finance", "E": "ESG", "R": "Risk"}


@dataclass
class World:
    names: list           # company tickers, same order as the rows of Z
    Z: np.ndarray         # (n_companies, dim), unit-length rows
    D: np.ndarray         # (n_agents, dim), unit-length rows (directions)
    G: np.ndarray = field(init=False)

    def __post_init__(self):
        self.G = self.D @ self.D.T

    def idx(self, ticker):
        return self.names.index(ticker)


# ----------------------------------------------------------------------------- full agent
def full_direction(D, beta):
    return np.asarray(beta) @ D


def full_scores(world, beta):
    return world.Z @ full_direction(world.D, beta)


# ----------------------------------------------------------------------------- one pair
def speaker_order(beta, G, agents=None):
    """Whiteboard rule: highest beta first; then each next speaker is the strongest challenger
    (most negative G) of the previous speaker among those who have not spoken. Ties -> lower index."""
    agents = list(range(len(beta))) if agents is None else list(agents)
    first = max(agents, key=lambda k: (beta[k], -k))
    order = [first]
    while len(order) < len(agents):
        last = order[-1]
        left = [k for k in agents if k not in order]
        order.append(min(left, key=lambda k: (G[last, k], k)))
    return order


def agent_margins(zA, zB, D, beta):
    return np.asarray(beta) * (D @ (zB - zA))


@dataclass
class PairResult:
    a: str
    b: str
    winner: str
    loser: str
    margins: np.ndarray        # per agent, in agent index order
    order: list                # speaker order (agent indices)
    running: list              # running evidence after each speaker
    final: float
    tie: bool
    full_diff: float           # S_ALL(B) - S_ALL(A)
    messages: int


def debate_pair(world, beta, ia, ib, order=None, agents=None):
    """Run the full debate for company ia (A) vs ib (B). All listed agents speak."""
    zA, zB = world.Z[ia], world.Z[ib]
    m = agent_margins(zA, zB, world.D, beta)
    order = speaker_order(beta, world.G, agents) if order is None else order
    running = np.cumsum([m[k] for k in order]).tolist()
    final = running[-1]
    tie = (final == 0.0)
    winner_is_b = final > 0
    a, b = world.names[ia], world.names[ib]
    d_all = full_direction(world.D, beta)
    return PairResult(a, b, b if winner_is_b else a, a if winner_is_b else b, m, order, running, final, tie,
                      float((zB - zA) @ d_all), len(order))


def check_identity(res, atol=1e-9):
    """Fail loudly if margins do not add up to the full agent's score gap."""
    if not np.isclose(res.margins.sum(), res.full_diff, atol=atol):
        raise AssertionError(f"identity broken for {res.a} v {res.b}: sum(m)={res.margins.sum()} "
                             f"vs S_ALL gap={res.full_diff}")


# ----------------------------------------------------------------------------- tournament
@dataclass
class Tournament:
    winner: str
    steps: list            # list[PairResult]
    rounds: list           # candidate sets after each debate
    messages: int


def tournament(world, beta, candidates, decide=None):
    """Queue method: pop two, debate, loser eliminated, winner goes to the back of the queue.
    `decide(ia, ib)` may be supplied to use a different decision rule (baselines); it must return
    a PairResult. Default = the full debate."""
    decide = decide or (lambda ia, ib: debate_pair(world, beta, ia, ib))
    queue = list(candidates)
    steps, rounds = [], [[world.names[i] for i in queue]]
    msgs = 0
    while len(queue) > 1:
        ia, ib = queue.pop(0), queue.pop(0)
        res = decide(ia, ib)
        steps.append(res); msgs += res.messages
        win = ia if res.winner == world.names[ia] else ib
        queue.append(win)
        rounds.append([world.names[i] for i in queue])
    return Tournament(world.names[queue[0]], steps, rounds, msgs)


# ----------------------------------------------------------------------------- baselines
def decide_majority(world, beta, ia, ib):
    """Each agent votes for the side its own direction prefers (sign of dz.d_k); majority wins.
    With 3 agents there is no tie unless a margin is exactly 0."""
    m = agent_margins(world.Z[ia], world.Z[ib], world.D, beta)
    votes_b = int((m > 0).sum())
    winner_b = votes_b > len(m) / 2 or (votes_b * 2 == len(m) and m[int(np.argmax(beta))] > 0)
    a, b = world.names[ia], world.names[ib]
    d_all = full_direction(world.D, beta)
    return PairResult(a, b, b if winner_b else a, a if winner_b else b, m, list(range(len(m))), [],
                      float(m.sum()), False, float((world.Z[ib] - world.Z[ia]) @ d_all), len(m))


def decide_single(world, k, ia, ib):
    """A single agent k alone (perspective-only ranking)."""
    m = np.zeros(len(world.D)); m[k] = (world.Z[ib] - world.Z[ia]) @ world.D[k]
    a, b = world.names[ia], world.names[ib]
    return PairResult(a, b, b if m[k] > 0 else a, a if m[k] > 0 else b, m, [k], [float(m[k])], float(m[k]),
                      False, 0.0, 1)


def top_pick_plurality(world, beta, candidates):
    """Each agent votes for ITS OWN top-scoring candidate; most votes wins. If all three differ,
    the highest-beta agent's pick wins (documented tie rule)."""
    picks = [candidates[int(np.argmax(world.Z[candidates] @ world.D[k]))] for k in range(len(world.D))]
    counts = {}
    for p in picks:
        counts[p] = counts.get(p, 0) + 1
    best = max(counts.values())
    leaders = [p for p, c in counts.items() if c == best]
    if len(leaders) == 1:
        return world.names[leaders[0]]
    return world.names[picks[int(np.argmax(beta))]]


# ----------------------------------------------------------------------------- early stopping
def _bounds(world, beta, ia, ib, spoken, margins, kind):
    """Interval [lo, hi] that the sum of the REMAINING agents' margins must lie in, given what has
    been said. kind:
      'loose' : |dz.d| <= 2 for unit vectors, so each remaining agent is bounded by 2*beta_j
      'cs'    : Cauchy-Schwarz with the actual distance: |dz.d_j| <= ||dz||, so beta_j*||dz||
      'tight' : uses G and what was said. Split dz into the part inside span(spoken directions) -
                fully known from the spoken margins - and the rest (only its length is known).
                The interval is then exact: no smaller interval is justified by the information."""
    remaining = [k for k in range(len(beta)) if k not in spoken]
    dz = world.Z[ib] - world.Z[ia]
    nz = float(np.linalg.norm(dz))
    bj = np.asarray(beta)[remaining]
    if kind == "loose":
        r = 2.0 * bj.sum(); return -r, r
    if kind == "cs":
        r = nz * bj.sum(); return -r, r
    G = world.G
    w_norm2 = float(bj @ G[np.ix_(remaining, remaining)] @ bj)
    if not spoken:
        r = nz * np.sqrt(w_norm2); return -r, r
    S = list(spoken)
    u = np.array([margins[k] / beta[k] for k in S])             # dz . d_s, recovered from the margins
    GSS = G[np.ix_(S, S)]
    Ginv = np.linalg.pinv(GSS)
    c = Ginv @ u
    dz_par2 = float(u @ Ginv @ u)
    dz_perp = np.sqrt(max(0.0, nz ** 2 - dz_par2))
    g = G[np.ix_(S, remaining)] @ bj                             # d_s . w
    centre = float(c @ g)
    w_perp = np.sqrt(max(0.0, w_norm2 - float(g @ Ginv @ g)))
    r = dz_perp * w_perp
    return centre - r, centre + r


def debate_pair_early_stop(world, beta, ia, ib, kind="tight"):
    """Same debate, but stop as soon as the remaining agents CANNOT flip the sign of the evidence.
    Returns a dict with the winner and the number of messages used. With a valid bound the winner
    is identical to the full debate (the stress test verifies this)."""
    order = speaker_order(beta, world.G)
    m = agent_margins(world.Z[ia], world.Z[ib], world.D, beta)
    spoken, run, b_wins = [], 0.0, None
    for k in order:
        spoken.append(k)
        run += m[k]
        if len(spoken) == len(order):
            b_wins = run > 0
            break
        lo, hi = _bounds(world, beta, ia, ib, spoken, m, kind)
        if run + lo > 0:
            b_wins = True
            break
        if run + hi < 0:
            b_wins = False
            break
    a, b = world.names[ia], world.names[ib]
    return {"a": a, "b": b, "winner": b if b_wins else a, "messages": len(spoken), "spoken": spoken}


# ----------------------------------------------------------------------------- trace printing
def format_pair_trace(world, beta, res, ia, ib, number=1):
    lines = [f"Debate {number}: A={res.a} vs B={res.b}      dz = z_B - z_A"]
    for step, k in enumerate(res.order, 1):
        why = "highest weight speaks first" if step == 1 else \
            f"strongest challenger of {AGENT_NAME[AGENTS[res.order[step - 2]]]} (G={world.G[res.order[step - 2], k]:+.2f})" \
            if step < len(res.order) else "remaining agent"
        if step > 1 and step == len(res.order) and len(res.order) > 2:
            why = "remaining agent"
        lines.append(f"  speaker {step}: {AGENT_NAME[AGENTS[k]]:8s} beta={beta[k]:.2f} [{why}]  "
                     f"margin {res.margins[k]:+.4f}  running evidence {res.running[step - 1]:+.4f} "
                     f"({'favours ' + res.b if res.running[step - 1] > 0 else 'favours ' + res.a})")
    lines.append(f"  final evidence {res.final:+.4f}  =>  {res.winner} wins, {res.loser} eliminated")
    lines.append(f"  identity check: m_F+m_E+m_R = {res.margins.sum():+.4f}  vs  "
                 f"S_ALL({res.b}) - S_ALL({res.a}) = {res.full_diff:+.4f}  "
                 f"{'OK' if np.isclose(res.margins.sum(), res.full_diff, atol=1e-9) else 'MISMATCH'}")
    return "\n".join(lines)


def format_tournament_trace(world, beta, t):
    out = []
    for i, res in enumerate(t.steps, 1):
        out.append(format_pair_trace(world, beta, res, world.idx(res.a), world.idx(res.b), i))
    out.append("Bracket: " + "  ->  ".join("{" + ", ".join(r) + "}" for r in t.rounds))
    out.append(f"Tournament winner: {t.winner}   (agent messages used: {t.messages})")
    return "\n".join(out)
