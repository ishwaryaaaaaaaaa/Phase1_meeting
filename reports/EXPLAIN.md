# EXPLAIN — how to talk about this

Written so you can read it cold and explain the project without me in the room. Every number here comes
from the actual run in `reports/RESULTS.md` — nothing is made up for the talk.

---

## 1. The 5-minute talk script

**The problem.** We're trying to get AI agents to pick the best company for an investor, but from
different angles at once — Finance, ESG, Risk — the way a real investment committee has specialists who
each look at one thing and then argue it out.

**The idea.** Represent each company as a point in space — an *embedding*, produced by feeding its SEC
filing text through a pretrained language model. Represent each perspective (Finance, ESG, Risk) as a
*direction* — an arrow in that same space, learned from real data on which companies score well on that
perspective. Projecting a company's point onto a direction gives that company's score on that
perspective. An investor is just a set of weights — how much they personally care about each
perspective — learned the same way a recommender system learns your taste: by asking you to pick between
pairs.

**Full agent vs. partial agents.** If you combine the three directions using the investor's weights, you
get one "full agent" that sees everything and can score any company directly. But in our setup, each
*partial* agent only knows its own one direction and weight — none of them individually knows what the
full agent knows. The question: can three partial agents, by debating, reach the same conclusion the
all-seeing full agent would reach?

**The pipeline, what each step produced.**
1. Picked 30 real US companies across 10 sectors, on purpose chosen to create disagreement between
   Finance, ESG and Risk (e.g. oil majors score well on Finance, badly on ESG).
2. Downloaded each company's actual SEC 10-K filing and pulled out the Business and Risk Factors
   sections — real text, nothing made up.
3. Built real labels for Finance (revenue growth + profit margin), ESG (a public risk-rating dataset),
   and Risk (3-year price volatility and drawdown) — all from named public sources.
4. Turned each company's filing text into one vector using a pretrained embedding model
   (`all-MiniLM-L6-v2`), sanity-checked it (same-sector companies cluster together far above chance).
5. Learned one direction per perspective from the labels, tested *honestly* — holding out whole
   companies, never just pairs — and got real, well-above-chance accuracy on all three (Finance 86%,
   ESG 84%, Risk 75% pair accuracy on held-out companies).
6. Built three investor profiles with different weights.
7. Ran the actual debate engine: watch one real debate happen, step by step, and a full 30-company
   tournament.
8. Checked the debate always agrees with the full agent (it has to, mathematically — see the "why" below),
   compared it against simpler baselines, and ran stress tests where that guarantee breaks down on
   purpose, to see what actually happens then.

**One real result to say out loud.** In this run, every investor's tournament lands on the same overall
winner (Microsoft) — but a Finance-heavy and an ESG-heavy investor's *full* 30-company rankings only
correlate at 0.57, meaningfully different priorities, that happened to agree at the very top. And under
noise, the full 3-agent debate is consistently more robust than simple majority voting — a real, testable
finding, not the identity.

---

## 2. Glossary — one line each

- **Embedding** — turning a piece of text into a list of numbers (a vector / point in space) that a
  model produces, such that similar meanings end up as nearby points.
- **Direction** — a single arrow (unit vector) in that same space, representing "more of this quality
  lies this way" for one perspective.
- **Projection (the dot product `z . d`)** — how far along a direction a point lies; this is the score.
- **Beta (β)** — the investor's own weights on the three perspectives, non-negative, summing to 1.
- **Full agent** — the agent that combines all three directions with the investor's weights and can
  score any company directly; it "sees everything."
- **Partial agent** — one of the three perspective agents; each knows only its own direction and weight.
- **Margin** — how much one agent's view favours company B over company A on one pair, in one round.
- **Running evidence** — the running total of margins as agents speak in turn; its final sign decides
  the winner of that pair.
- **Tournament** — repeatedly debating pairs from a queue, eliminating the loser, until one company
  remains.
- **Cosine (similarity)** — a number between −1 and 1 measuring how aligned two vectors are, independent
  of their length; used for both "how similar are two companies" and "how aligned are two directions."
- **G matrix** — cosine similarity between every pair of perspective directions; tells you which agents
  tend to agree or challenge each other (not which companies are similar).

---

## 3. What I did, step by step (read this aloud)

1. "I picked 30 real companies from 10 different industries, on purpose including some that are strong
   financially but weak on ESG, and vice versa, so there'd be real disagreement to debate."
2. "For each one, I downloaded their actual SEC filing and pulled out two sections: what the business
   does, and what risks it discloses."
3. "I built three scores for each company from public data: financial growth and profitability, an ESG
   risk rating, and market risk from three years of price history."
4. "I turned each company's filing text into a 384-number vector using a pretrained AI embedding model
   — the same kind of model behind semantic search."
5. "I checked that worked by looking at which companies ended up near each other — same-sector companies
   cluster together far more than random chance, which tells me the embedding is capturing something
   real."
6. "I learned one direction per perspective — Finance, ESG, Risk — by training a model to rank company
   pairs the way the real labels do, and tested it honestly by holding out whole companies it had never
   seen."
7. "I built three investor types with different weights on those three perspectives."
8. "I ran the actual multi-agent debate: each perspective agent argues using only its own view, they
   take turns, and I track the running evidence until the sign decides a winner — then I checked, by
   hand and in code, that this always matches what a single all-seeing agent would have said."
9. "Then I deliberately broke that guarantee — added noise, removed an agent, let agents stop early — to
   see what actually happens when the setup isn't perfectly clean, since the clean case is guaranteed to
   work by the math, not by anything interesting about debate."

---

## 4. Q&A — honest answers

**Q: Why these three perspectives (Finance, ESG, Risk)?**
They came from the professor/Kunal's whiteboard notes as the three the project was asked to build. They
also naturally pull against each other in real investing (growth vs. safety vs. responsibility), which
is what makes a debate meaningful instead of all agents trivially agreeing.

**Q: Why are the directions shared across every investor — only the weights change?**
That's how the whiteboard set it up (one `d_F, d_E, d_R` box, separate beta box per investor), and it
makes sense if you think of "what makes a company strong on Finance" as an objective fact independent
of who's asking, while "how much I care about Finance" is personal. **I have not verified this
assumption against any paper or further discussion — it's the most important thing to confirm.**

**Q: Where do the labels come from, and why is that a limitation?**
Finance and Risk are pulled live from Yahoo Finance today. ESG comes from a public snapshot dataset
dated 2024-06-07, because Yahoo Finance's own live ESG data field returned **empty for every single
company I tested** — a real, current gap in the free API, confirmed directly, not assumed. That means
the ESG numbers are up to ~2 years older than the other two labels — a genuine inconsistency worth
flagging, not hiding.

**Q: Why does the debate winner always equal the full agent's winner?**
Because the scores are built *linearly*: the full agent's score gap between two companies is
*literally* defined as the sum of the three agents' margins (`m_F + m_E + m_R = S_ALL(B) − S_ALL(A)`,
verified on every single pair debated this run — zero mismatches). So once all three agents have spoken,
they **cannot** disagree with the full agent — it's arithmetic, not evidence that debate is a good idea.
**This is the central research-gap point**: the clean demo proves the *pipeline* works end-to-end, not
that debate beats a single model. The interesting question only shows up once you break the clean case —
which is exactly what Phase 8's stress tests do.

**Q: So what did the stress tests actually show?**
Two real findings: (1) under added noise, the full 3-agent debate is consistently more accurate than
simple majority voting at every noise level tested; (2) early stopping (letting agents skip speaking
once the outcome can't change) saves up to a third of agent messages for some investors, but saves
*nothing at all* for the "balanced" investor, because its weights are close enough together that the
running evidence rarely locks in early — a real, investor-dependent effect, not a bug.

**Q: Why is G not about which companies are similar?**
G is the cosine similarity *between the three directions themselves* — it tells you whether, say, the
Finance and Risk agents tend to see the world the same way (positive G) or oppose each other (negative
G). It says nothing about any individual company; the company-level comparison is a completely separate
step (`Δz = z_B − z_A`, projected onto each direction).

**Q: Why cosine instead of covariance for G?**
Covariance is unbounded and changes with the scale of the vectors; cosine is bounded between −1 and 1 and
only measures direction, which is what "do these two agents tend to agree" actually means.

**Q: Only 30 companies and a 384-dimensional embedding — isn't that a problem?**
Yes, and I say so directly in the results: the directions are only weakly identified at this scale — the
regularisation strength matters a lot (see the sensitivity plot), bootstrap resampling shows real but
not huge instability (cosine 0.77–0.87 with the full-data direction), and a plain "predict from sector
alone" baseline already gets most of the way there for Finance and ESG. More companies would sharpen
all of this.

**Q: What should happen next?**
Scale the company set up toward the brief's 30+ target with more variety, get a second, truly
finance-specialised embedding model if one becomes available, run the interactive investor questionnaire
with a real person (built, ready, not yet run — see below), and push the stress tests further (e.g. does
the noise-robustness advantage of debate over majority vote hold with more agents, not just three?).

---

## 5. Things I inferred and need to confirm with the professor / Kunal

1. **Speaker order rule** — "highest weight speaks first, then the strongest challenger (most negative
   G) of whoever just spoke, repeat" is my reading of the whiteboard's arrows and labels ("strongest
   challenger"); the notes don't spell out a formal rule, and I could not check this against
   `debate_prototype.py` because that file was not present on this machine (see below).
2. **Directions shared across investors, only beta differs** — matches the whiteboard's layout (one
   direction box, separate beta per investor) but is not explicitly stated as a modelling choice
   anywhere in the notes.
3. **Meaning of "I001" / "I002"** — my reading is two different investors (different beta) debating the
   same company set to see if they land on different winners. Not confirmed; could mean something else
   entirely (e.g. two specific real investors, or two specific beta presets the group already agreed on).
4. **Meaning of "A2A messages → modification"** — unclear from the notes. Not implemented, beyond the
   debate's existing message-passing (margins/running evidence); flagged rather than guessed at.
5. **The 0.81/0.91 vs. 0.15 inconsistency on the whiteboard** — the notes give `S_ALL(A)=0.81`,
   `S_ALL(B)=0.91` (a gap of 0.10) in one place, but the worked margins add up to 0.15 in another. Under
   the model these two numbers must be equal (that's the whole identity), so one of the two board
   examples is illustrative only, not a single consistent worked example. I did not silently fix this —
   the test suite (`tests/test_engine.py`) reproduces the margins/running-evidence numbers (0.12, 0.08,
   0.15) exactly, and separately reproduces the bracket/ranking example (B>C>A>D,
   {A,B,C,D}→{B,C,D}→{B,C}→B) exactly, as two *different* worked examples rather than forcing them to be
   the same one.
6. **`debate_prototype.py` and `HANDOFF_for_new_chat.md` were not found** anywhere on this machine
   (searched the full user profile). I rebuilt the engine from the brief and the whiteboard photos
   instead of porting the original prototype — the engine's logic was still cross-checked against the
   whiteboard's exact numbers (see `tests/test_engine.py`), so I'm confident it's faithful to what's on
   the board, but it may differ from the original file in code structure or an implementation detail the
   file contained that the photos didn't show.
7. **The investor questionnaire has not been run with a real person.** `p06_investor.py --interview` is
   built, tested for syntax, and ready — it picks the 18 company pairs where the three perspectives
   disagree the most, so the answers actually reveal something about weights. Run it when you're back:
   `cd src && python p06_investor.py --interview`, then `python p07_debate.py && python p08_stress.py`
   to fold your own profile into every report.
