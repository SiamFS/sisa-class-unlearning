# Does the gate need retraining?

Component audit of the learned router in the SISA class-unlearning system.
CIFAR-10 · 2 shards · ResNet-20 specialists · 77,522-parameter gate · deleted class: `cat`

---

## Verdict

| Question | Answer |
|---|---|
| Can an attacker extract data from the gate? | **No.** Three attempts, nothing recoverable. |
| Can we skip the retrain and still claim *exact* unlearning? | **No.** Exactness is a lineage property, not an attack-resistance property. |

Both are true at once. The measurements are real and worth publishing — they just answer
the security question, not the exactness one.

---

## 1. The distinction that keeps getting merged

The gate maps an image to a shard index. It never outputs a class label. That makes it
tempting to conclude it holds nothing worth deleting — and on the **security** question,
that conclusion is correct.

But "exact unlearning" is not a security claim:

> A system is exactly unlearned if its parameters are reproducible by training on
> `D \ D_cat` — the dataset with the deleted class removed.
> — Cao & Yang 2015; Bourtoule et al. 2021

Nothing in that definition mentions an adversary. An auditor checks data lineage, not
attack success. A gate trained on data containing cats is, by inspection, not reproducible
from a dataset without them — so it fails the definition even though every attack below
also failed.

---

## 2. Evidence

### EXP 1 — Confidence-threshold membership inference · *no leakage found*

Compare the gate's routing confidence on cat images it trained on against held-out cats.

```
trained-on cats (n=1000)   shard-2 confidence  0.9673
never-seen cats (n=1000)   shard-2 confidence  0.9325
MIA AUC                                        0.5110   (0.5 = chance)
```

Chance. **Stated limit:** this is the weakest class of membership attack, and the gate emits
one bit per query. A null result here is weak evidence, not proof. Shadow-model / LiRA-style
attacks were not attempted.

### EXP 2 — Gate–specialist contradiction · *INVALID, discarded*

The idea: after deletion the gate confidently says "shard 2" while the specialist cannot
classify the image; that disagreement fingerprints the deleted class. It appeared to work —
cat's gap was 3.3× the next class.

**The result was an artifact.** Deletion was simulated by masking cat's output column on a
specialist still *trained* on cat. Its internal cat features were intact; only the output was
blocked, so the low confidence came from reading the runner-up logit, not from unlearning.
A genuinely retrained specialist never builds those features and would likely classify cats
as dog with high confidence — no contradiction at all.

Recorded rather than deleted: it demonstrates why simulated deletion cannot stand in for a
real one, which applies to any attack evaluated before an actual unlearning run.

### EXP 3 — Linear probe on gate features, with random-init control

Does the gate's internal representation contain anything cat-specific? Its supervision for a
cat image is the label *shard 2* — identical to bird, deer, dog, frog, horse. Cat is never
individually identified during training.

Probe: linear classifier on penultimate features, detecting cat among shard-2 test images
(n = 6,000, 60/40 stratified split).

| Model | Cat-detection AUC |
|---|---|
| Random-init gate (saw no data at all) | 0.6334 |
| Trained gate (saw cat images) | 0.7204 |
| Specialist (trained on class labels) | 0.9704 |
| **Attributable to training** | **+0.0870** |

Cats are distinctive enough that an untrained network already separates them at 0.63. Only
+0.087 is attributable to having seen data containing cats — and it was measured on *held-out*
images, so it reflects generalizable visual structure, not memorization of training samples.

### Summary of extraction attempts

| Attack goal | Result | Interpretation |
|---|---|---|
| Reconstruct cat images | Infeasible | One bit of supervision per sample, shared across six classes. Nothing per-image to invert. |
| Identify a specific training image | AUC 0.511 | Chance. No membership signal. |
| Detect that cat existed | +0.087 AUC | Marginal, and generalizable structure rather than memorization. |
| Identify which class was deleted | Test invalid | Needs a real unlearning run to evaluate. |

---

## 3. Efficiency: if the gate retrains every time, what is left of the speedup?

Measured against `logs/training_20260827_021813.log`:

| Component | Cost | Share of a deletion |
|---|---|---|
| Gate retrain — full, all data, every deletion | 49.35 s | **6.2%** |
| Specialist retrain — shard 2, slices 2–6 | 246 epochs | **93.8%** |

**The gate is 6.2% of a deletion round.** It is not the bottleneck, and removing it buys back
almost nothing.

The real cost is structural: deleting cat forces **58.6% of all specialist training**
(246 of 420 epochs) to be redone — because cat sits at slice 2 of 6 in the larger shard, and
every slice after it must retrain.

That is the number to attack, and it has nothing to do with routing. Slice *position*
determines deletion cost: a class in the last slice costs almost nothing to remove, one in
the first costs the whole shard. Ordering slices by deletion likelihood is the established
fix (ARCANE §3.4 deletion-likelihood sorting).

**Caveat on the arithmetic:** a clean speedup figure against a monolithic retrain cannot be
quoted yet — the monolithic baseline has only ever run for 2 epochs as a smoke test
(`monolithic_20260827_020518.log`, 32.46 s, 59.4% accuracy). The denominator for every
efficiency claim in the paper does not exist until that run completes at full budget.

### Why the gate still caps you eventually

Specialist cost falls as shard count K grows; gate cost does not, since the router always
trains on everything. That makes it a fixed serial fraction — Amdahl's law applied to
unlearning — so speedup asymptotes rather than scaling indefinitely. Invisible at K=2 and
6.2%; at large K the router becomes the limit. Slicing the gate the way specialists are
sliced removes that ceiling, and is the natural future-work extension.

---

## 4. Terminology: there is no "approximate exact" unlearning

The two terms name mutually exclusive guarantees. Pairing them reads as hedging, and
reviewers will treat it as a claim the authors could not commit to.

- **Exact** — the deployed weights are reproducible by training on `D \ D_cat`. Retrain the
  gate and this is simply true. Call it exact, no qualifier.
- **Approximate** — the weights are not reproducible, and the privacy claim rests on empirical
  evidence such as MIA results. A legitimate, well-populated category — just a different one,
  with different baselines (SCRUB, Amnesiac, Fisher forgetting).

If a hybrid ever needs describing, use explicit phrasing rather than a compound term:
*"exact unlearning of the specialist ensemble, with an approximately unlearned router."*
At 6.2% of a deletion round, that sentence costs more in reviewer confidence than the retrain
costs in compute.

---

## 5. The decision

Since no attack succeeded, this is about which claim the paper makes — not about defending
against a demonstrated threat.

| Position | Gate handling | System accuracy | What it obligates |
|---|---|---|---|
| **Exact** *(recommended)* | Retrain every deletion | 87.76% | Nothing further. The guarantee is structural. |
| **Gate-free** | No router exists | 76.38% | Nothing to retrain, still exact. Costs 11.4 points. |
| **Hybrid** | Keep gate, don't retrain | 87.76% | Exact specialists, approximate router. Invites comparison against SCRUB / Amnesiac / Fisher — all cheaper. |

Avoid the hybrid. It surrenders the exactness claim that distinguishes SISA-derived work, and
the saving is 6.2% of one deletion.

### Recommendations

1. **Keep the retrain.** 6.2% of a deletion round buys an unconditional claim instead of an
   empirical argument.
2. **Report gate-free routing as an ablation.** The 11.4-point gap quantifies what routing
   convenience costs in an exact-unlearning system. Four independent gate-free methods landed
   11–25 points short, so the gap is structural, not a tuning failure.
3. **Publish the residual-information measurement.** The +0.087 should fall to ~0 once the gate
   is retrained without cat. "Residual class information in the router, before vs. after
   retraining" honestly demonstrates that unlearning works — without requiring an attack to exist.
4. **Spend efficiency effort on slice ordering, not the router.** The gate is 6.2% of a deletion;
   slice position accounts for the other 93.8%.

---

## 6. Still unmeasured

- Every number here comes from a system in which **no class has actually been deleted yet**.
  They characterize what the un-retrained gate holds — not what a real unlearned system leaks.
- `experiments/exactness_eval.py` has never run at full budget. It is the paper's central claim
  and the prerequisite for evaluating any attack against a genuinely unlearned system.
- `experiments/monolithic_baseline.py` has only run 2 epochs. It is the denominator for every
  efficiency and accuracy claim.
- Stronger membership attacks (shadow-model, LiRA) were not attempted. The 0.511 is from the
  weakest attack class and should be described that way.
- The before/after residual figure in recommendation 3 needs one gate retrained without cat.
