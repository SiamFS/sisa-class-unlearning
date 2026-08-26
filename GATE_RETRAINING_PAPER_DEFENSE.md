# Defending router retraining in the paper

How to write up the fact that the learned router is retrained on every deletion.

Companion to [GATE_RETRAINING_ANALYSIS.md](GATE_RETRAINING_ANALYSIS.md), which holds the
measurements this document argues from.

---

## The precise claim

"Gates must be retrained" is too broad. The defensible statement distinguishes two cases:

| Router type | Retrain on deletion? | Why |
|---|---|---|
| **Learned** (MoE-style gate) | **Yes, every time** | Its weights are a function of the training data. Delete part of that data and the weights are no longer reproducible from what remains. |
| **Non-learned** (MSP self-routing) | **No, never** | It has no parameters of its own — it reads the specialists' softmax, and those are already retrained. Nothing to unlearn. |

So the claim to make is not "routers must be retrained." It is:

> A learned router buys **+11.4 points** of accuracy (76.38% → 87.76%) and costs **6.2%** of a
> deletion round. Both halves of that tradeoff are ours to state explicitly.

The only thing that reduces "every time" is slicing the router the way specialists are sliced,
so a deletion rewinds to the affected router slice instead of restarting. That is future work,
not a current property of the system.

---

## Five moves

### 1. Make it a contribution, not an apology

SISA (Bourtoule et al. 2021) aggregates by majority vote or averaged posteriors — no learned
router — and pays for it in accuracy. Adding one is the delta. Frame it that way:

> *We show that a learned router substantially improves aggregation accuracy (+11.4 points over
> confidence-based self-routing), and quantify the exactness cost it introduces: because the
> router observes every shard's data, it must be retrained on each deletion, adding 6.2% to a
> deletion round.*

This positions the work as the first to **quantify the accuracy-vs-exactness tradeoff of learned
routing in exact unlearning**. Papers that add a router and say nothing about it have an
incomplete guarantee; you close the hole and price it.

### 2. Pre-empt the completeness attack

The sharpest reviewer question is:

> *"You retrain the specialists, but the router still saw the deleted class. Your system is not
> exactly unlearned."*

If they raise it, the paper is in trouble. If you raise it first and answer it, it reads as rigour.
State plainly that exactness is a property of the **whole deployed system**, and that any component
trained on deleted data is in scope.

### 3. Pre-empt the Amdahl objection

> *Router cost is invariant in the number of shards K, while specialist cost falls as K grows; the
> router therefore becomes the serial bottleneck at large K. At K = 2 it is 6.2% of a deletion
> round. Slicing the router as the specialists are sliced would remove this ceiling and is left to
> future work.*

Naming your own limitation with the correct formalism (a fixed serial fraction — Amdahl's law) is
worth more than hoping nobody notices.

### 4. Use the null results honestly — they become an asset

**Do not write** "we retrain the router because an attacker could otherwise extract the deleted
class." Two attempts to build that attack failed (see the analysis doc); a reviewer may fail too,
and then the stated motivation collapses.

Write this instead:

> *We retrain the router because exactness is a property of parameter lineage, not of empirical
> attack resistance. Consistent with this, a non-retrained router leaks little in practice:
> membership inference on the router alone achieves 0.511 AUC, and a linear probe recovers only
> +0.087 AUC of class information above a randomly initialised control. We nonetheless retrain,
> because a guarantee that depends on the failure of the attacks we happened to try is an
> approximate guarantee.*

That paragraph shows you measured rather than assumed, distinguishes exact from approximate
correctly, and demonstrates that a negative security result does not discharge a definitional
obligation. It is the difference between asserting exactness and understanding it.

### 5. Redirect the efficiency discussion to where the cost actually is

If a reviewer says retraining the router undermines the efficiency claim, answer with the
measurement:

- Router: **6.2%** of a deletion round (49.35 s)
- Slice position: **93.8%** — deleting `cat` redoes **58.6%** of specialist training (246 of 420
  epochs) because it sits at slice 2 of 6

That converts the objection into your future-work section on deletion-likelihood-ordered slicing
(cf. ARCANE §3.4).

---

## Anticipated reviewer questions

| Question | Answer |
|---|---|
| Isn't the router a monolithic component in a system designed to avoid them? | Yes — stated as a limitation, with the sliced-router extension as the fix. |
| Why not skip the router retrain, since no attack succeeds? | Then the guarantee is approximate, and the paper owes comparisons to SCRUB, Amnesiac and Fisher forgetting — all cheaper. The saving is 6.2%. |
| Does retraining the router negate the speedup? | No: 6.2% of a deletion. Slice ordering dominates at 93.8%. |
| Can you call it "approximate exact" unlearning? | No — mutually exclusive guarantees; the compound reads as hedging. Retrain and call it exact, no qualifier. |
| Why is the router not sharded like the specialists? | Honest answer: it isn't, and that is the design's weak point. Name it before a reviewer does. |

---

## Terminology

- **Exact** — deployed weights are reproducible by training on `D \ D_forget`. Retrain the router
  and this is simply true. Use no qualifier.
- **Approximate** — weights are not reproducible; the privacy claim rests on empirical evidence.
  A legitimate category, with different baselines.
- **"Approximate exact"** — not a thing. If a hybrid ever needs describing, spell it out:
  *"exact unlearning of the specialist ensemble, with an approximately unlearned router."*

---

## Before any of this goes in a paper

Every number above comes from a system in which **no class has actually been deleted yet**.

1. `python experiments/exactness_eval.py --class-name cat` — the central claim, never run at full
   budget.
2. `python experiments/monolithic_baseline.py` — the denominator for the 6.2% and for every
   accuracy comparison. Currently only a 2-epoch smoke test (32.46 s, 59.4%).

Until both run, the defense above is sound in structure but unsupported in its constants.
