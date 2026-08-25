# SISA Accuracy Recovery — Implementation Plan (W18–W21)

**Extends** `IMPLEMENTATION_PLAN.md` (W1–W17). Same numbering convention, same non-negotiables:
exact unlearning, reproducibility (global seed), dataset/architecture agnostic, never tune on test.

**Starting point (measured, 2026-08-26):** final SISA system accuracy **70.35%** on the CIFAR-10
test set, 2 shards × 5 slices, gating-network routing.

---

## 1. Diagnosis (measured, not assumed)

Decomposition from the per-shard confusion matrices and `experiments/gate_free_routing_probe.py`:

| Quantity | Value |
|---|---|
| Shard 1 alone (4 vehicle classes) | 82.1% |
| Shard 2 alone (6 animal classes) | 68.9% |
| **Oracle ceiling** (perfect routing, current specialists) | **74.15%** |
| Gating network routing accuracy | 94.41% |
| **Actual system** | **70.35%** |

Two independent losses, and they are not the same size:

- **Routing: ~3.8 points.** `combined ≈ routing_acc × 74.15%` holds across every strategy measured.
- **Within-shard classification: ~16 points.** Specialists should reach ~90% i.i.d. on 4/6 CIFAR
  classes; they reach 82.1%/68.9%.

### 1.1 Root cause of the within-shard loss: task-recency bias

Class-sequential slicing makes each shard a class-incremental learning problem. Recall lines up
exactly with slice order in both shards:

- Shard 1 (airplane → automobile → ship → truck): 74.9, 82.2, 78.0, **93.3**
- Shard 2 (bird → cat → deer → dog → frog → horse): 66.8, 56.5, 54.6, 55.6, **87.2, 92.4**

Precision/recall confirms the mechanism: horse 0.91 recall / 0.55 precision (over-predicted),
airplane 0.61 / 0.85 (under-predicted). Late classes swamp early ones.

**Cause — the replay batch math.** `train_model.py` splits every batch as
`main = int(batch_size * (1 - replay_ratio))`, with `REPLAY_RATIO` fixed at 0.3. At shard 2's
slice 5 that gives, per 64-sample batch: horse ≈36.7, frog ≈11.3, and bird/cat/deer/dog **4 each**.
Fair share is 10.7. Horse gets 3.4× its share, each old class 0.375× — a **9:1 imbalance**, in
every batch. The buffer itself is fine (1000/class stored, cap working); only 20 of 64 slots are
allocated to all old classes combined.

### 1.2 Second cause: augmentation never runs

`check_class_balance_and_augmentation` returns `None` when `balance_ratio >= 0.95`. Class-isolated
sharding plus equal slices makes CIFAR-10 shards *perfectly* balanced by construction, so it always
takes that branch. Both shards logged "Classes are perfectly balanced - NO AUGMENTATION".

Three separate defects here:
1. Augmentation is treated as an imbalance remedy, not a regularizer, so it never runs.
2. No `RandomCrop` exists in any augmentation config — only flip and color jitter.
3. `train_transforms(batch_x)` is applied to a **batched** tensor, so torchvision v1 transforms make
   one random decision for all 64 samples rather than per-sample.

### 1.3 Third cause: degenerate single-class validation

Shard 1's slice 1 contains only airplane. Validation is filtered to 1 active class → the filtered
output has one column → softmax is trivially 1.0 → cross-entropy is **exactly 0.0**. Early stopping
can never improve on zero, so after `patience` epochs it restores the **epoch-0** weights, discarding
every subsequent epoch. Log: `Best val_loss: 0.000000 at epoch 0 / Restored best model from epoch 0`.

### 1.4 Routing: what was measured

`experiments/gate_free_routing_probe.py` evaluates 12 post-hoc routing scores on the existing
checkpoints (no retraining). Results:

| strategy | kind | routing | combined |
|---|---|---|---|
| oracle | ceiling | 100.00% | 74.15% |
| gating network (W16) | learned | 94.41% | 70.35% |
| **ncm_cosine** | free | **75.06%** | **57.62%** |
| mahalanobis | free | 71.59% | 54.23% |
| zscore_softmax | free | 66.60% | 55.33% |
| energy | free | 65.58% | 53.15% |
| max_logit | free | 65.05% | 54.13% |
| max_softmax (current fallback) | free | 61.44% | 52.22% |
| ncm_euclidean | free | 60.55% | 41.93% |
| neg_entropy | free | 58.45% | 50.11% |

**Conclusions.** (a) Feature-space prototype scores beat every logit-space score by ~10 points,
confirming and extending W15 (which only tested logits). (b) No post-hoc score closes the 12.7-point
gap to the gate. (c) The specialists were trained with plain cross-entropy, which optimizes for
*separability*, not *compactness* — so no "not mine" region exists in feature space to detect.

### 1.5 Gating network defects (if it is kept)

- **60/40 class imbalance, uncorrected.** `nn.CrossEntropyLoss()` with no weights, on 27,000 shard-2
  labels vs 18,000 shard-1. Result: shard 1's classes leak out at ~11% (airplane 20%), shard 2's at
  ~2%; routing distribution 3679/6321 against a true 4000/6000. **This imbalance was introduced by
  W17** — the pre-W17 even/odd split was 5+5 classes, perfectly balanced.
- **~544k parameters** to make a binary decision, ~96% of it in `fc1 = Linear(64*8*8, 128)`. A full
  specialist is ~620k.
- `random_state=42` hardcoded instead of `config.SEED` (harmless only because SEED == 42 today).
- Only `RandomHorizontalFlip`, no crop, and the same batch-level transform defect as §1.2.

---

## 2. Workstreams

### W18 — Class-balanced replay [Phase 1, largest single effect]

- **Files:** `training/replay_buffer.py` (new `compute_replay_ratio`), `training/entry_training.py`,
  `unlearning/sisa_unlearning.py` (`_retrain_shard_incrementally`), `config.py`.
- **Change:** replace the fixed `config.REPLAY_RATIO` with a ratio derived from the class split,
  `n_old / (n_old + n_new)`, clamped at `REPLAY_RATIO_MAX`. Deterministic (a pure function of data
  composition — no RNG), so exactness is unaffected.
- **Critical:** the identical helper must be used by the unlearning retrain path. If original
  training and unlearning retrain use different replay recipes, the W8 scratch-reference comparison
  is invalid and the exactness proof breaks.
- **W18e — batch stride bug (found during implementation, pre-existing).** The epoch loop advanced by
  `batch_size` while consuming only the first `main_batch_size = int(batch_size*(1-replay_ratio))`
  indices of each chunk, so the remainder of every chunk was silently dropped each epoch: **31% of
  each slice at the old 0.3 ratio, and ~72% at W18's balanced ratios** — which would have undercut
  the fix entirely. The loop now strides by `main_batch_size`, so every current-slice sample is used
  exactly once per epoch and replay tops each batch up to `batch_size`. Epoch loss is normalized by
  the actual step count. Measured in isolation on shard 2 slice 5: **2-epoch validation accuracy
  0.4830 → 0.5743.**
- **Acceptance:** at shard 2 slice 5 the ratio is 0.714, not 0.3; per-class batch share goes from
  4 to ~9.2 slots per old class (horse:bird from 9.2:1 to ~1.6:1); steps/epoch 85 → 300.

### W19 — Real augmentation, applied per sample [Phase 1]

- **Files:** `training/augmentation.py` (new), `training/train_model.py`, `training/entry_training.py`,
  `config.py`.
- **Change:** (a) add a `baseline` augmentation level applied regardless of class balance;
  (b) add `random_crop_padding` (the standard CIFAR `RandomCrop(32, padding=4)`); (c) implement
  crop + flip as **per-sample** batched tensor ops with a seeded `torch.Generator`, replacing the
  batch-level torchvision path.
- **Acceptance:** augmentation runs on perfectly balanced shards; two samples in one batch can
  receive different crops/flips; same seed reproduces identical augmented batches.

### W20 — Early stopping under degenerate validation [Phase 1]

- **Files:** `training/train_model.py`.
- **Change:** when fewer than 2 classes are active, validation loss is structurally 0.0 and carries
  no signal — fall back to monitoring training loss for early stopping and log the fallback.
- **Acceptance:** shard 1 slice 1 no longer restores epoch 0; the restored epoch is > 0.

### W21 — Cosine classifier: routing replacement + recency-bias mitigation [Phase 2]

- **Files:** `training/create_model.py` (new `CosineClassifier`, `penultimate()`), `plots.py`
  (`_run_sisa_batch` cosine routing mode), `config.py`, `experiments/gate_free_routing_probe.py`.
- **Change:** replace the final `Linear` with a scaled cosine classifier (LUCIR-style):
  `logits = s · cos(f, w_c)` with L2-normalized features and class vectors. This
  1. gives a **calibrated routing score for free** — `max_c cos(f, w_c)` over owned classes, which is
     exactly the `ncm_cosine` that led the probe at 75%, but with *learned* class vectors instead of
     post-hoc feature means; and
  2. removes the weight-magnitude growth that drives task-recency bias (LUCIR's original purpose),
     so it attacks §1.1 as well.
- **Exactness:** strictly simpler than the gate. The class vectors *are* ordinary final-layer weight
  rows, so they are already checkpointed, and W6's `_resize_model_head` drops row *c* with no new
  logic. There is **no separate router to retrain** — the affected shard's retrain rebuilds its own
  routing signal, and unaffected shards never saw *c* under class-isolated sharding. W16's
  `train_gating(excluded_classes=...)` step becomes unnecessary if routing switches to cosine.
- **Acceptance:** `CosineClassifier` exposes `in_features`/`out_features` and registers its weight at
  `fc_layer.5.weight` so existing W6/loader code is untouched; cosine routing selectable via
  `config.ROUTING_MODE`; probe re-measures routing accuracy.

### W22 — Gating network fixes [Phase 3, conditional]

Only needed if W21's cosine routing does not reach the gate's 94.41%.

- **Files:** `training/train_gating_model.py`, `training/create_model.py` (`GatingNetwork`), `config.py`.
- **Change:** (a) inverse-frequency class weights in the loss, correcting the W17-introduced 60/40
  imbalance; (b) global average pooling instead of the 8×8 pool, shrinking the gate from ~544k to
  ~28k parameters; (c) `config.SEED` instead of the hardcoded `42`; (d) crop augmentation.
- **Acceptance:** routing distribution approaches the true 4000/6000 split; parameter count drops
  ~19×; routing accuracy does not regress.

---

## 3. Deferred (explicitly not in this pass)

- **W23 — architecture upgrade.** The 74.15% oracle ceiling is set by the specialists. A from-scratch
  CIFAR ResNet would lift it (ARCANE uses Wide ResNet on CIFAR-10). Deferred so W18–W21's effect can
  be measured cleanly against a fixed backbone. Must stay from-scratch — a pretrained backbone
  reintroduces the Lane B caveat rejected in §3.3 of the main plan.
- **W24 — re-tune and freeze.** `best_config.json` is stale twice over: measured under the old
  self-routing objective *and* under the broken replay/no-augmentation regime. Tuning must come last
  or it bakes the bugs into the hyperparameters.
- **W8 harness reconciliation.** `experiments/scratch_reference.py` and `exactness_eval.py` must match
  whatever routing survives Phase 3 before any exactness claim is made. Already flagged stale for the
  W16 gate in the main plan.

## 4. Rejected, with reasons (do not re-litigate)

| Option | Reason |
|---|---|
| Remove the gate with no replacement | −12 points, and *increases* inference compute (K specialists vs gate + 1) |
| Energy / logsumexp routing | Re-confirmed dead: 65.58% routing in the probe, vs 61.44% baseline |
| `ncm_euclidean` | 41.93% combined, worst measured — feature norm dominates the distance |
| Platt scaling per shard | 49.16% combined, and it introduces fitted parameters for no gain |
| Semantic clustering (W17) | Already applied to the real project — the log's class-to-shard map is the vehicles/animals split |
| Pure ARCANE one-class-per-submodel | Collapses the project's novelty (multi-class shards + class-sequential slicing) into ARCANE's |
| Pretrained backbone (Lane B) | Already rejected in main plan §3.3 for the ImageNet-prior exactness caveat |

---

## 5. TODO — all implemented, not yet run end-to-end

- [x] **W18a** `compute_replay_ratio` helper in `training/replay_buffer.py`
- [x] **W18b** config: `REPLAY_RATIO_MODE='balanced'`, `REPLAY_RATIO_MAX=0.8`
- [x] **W18c** wire into `training/entry_training.py`
- [x] **W18d** wire into `unlearning/sisa_unlearning.py::_retrain_shard_incrementally`
- [x] **W18e** batch stride bug in `train_model.py` (see above)
- [x] **W19a** `training/augmentation.py` — per-sample crop + flip, seeded generator
- [x] **W19b** config: `baseline` level + `random_crop_padding` on every level
- [x] **W19c** `train_model.py` uses the per-sample augmenter
- [x] **W19d** `entry_training.py` + `sisa_unlearning.py` return baseline augmentation when balanced
- [x] **W20** degenerate-validation fallback in `train_model.py`
- [x] **W21a** `CosineClassifier` in `training/create_model.py`
- [x] **W21b** `penultimate()`/`routing_cosine()` + checkpoint head-type detection in the loader
- [x] **W21c** `_route_via_cosine` in `plots.py` + `config.ROUTING_MODE`
- [x] **W21d** `head_cosine` row in `experiments/gate_free_routing_probe.py`
- [x] **W22a** inverse-frequency class weights in `train_gating_model.py`
- [x] **W22b** `GATING_POOL_SIZE` in `GatingNetwork` (544,258 → 28,162 params, 19.3×)
- [x] **W22c** `config.SEED` fix + per-sample crop augmentation for the gate

### Verified during implementation

| Check | Result |
|---|---|
| All touched files compile | pass |
| Replay ratio, shard 2 slices 1→5 | 0.000, 0.500, 0.600, 0.667, 0.714 (was 0.3 throughout) |
| Batch composition, shard 2 slice 5 | 18 current + 46 replay over 5 classes = 9.2/class (was 44 + 20 = 4/class) |
| Steps per epoch, shard 2 slice 5 | 300 (was 85, discarding 46 of every 64 samples) |
| Per-sample augmentation | different crops/flips within one batch; identical under the same seed |
| Degenerate validation (shard 1 slice 1) | monitors train loss, restored epoch 3 (was epoch 0) |
| W6 head resize with cosine head | 29/30 tensors transplanted, only `fc_layer.5.weight` reinitialized |
| Old linear checkpoints under `CLASSIFIER_TYPE='cosine'` | load correctly — checkpoint wins over config |
| Old 8×8 gate checkpoint under `GATING_POOL_SIZE=1` | loads correctly — pool size inferred from `fc1` |
| All three routing modes run | gating / cosine / confidence |

## 5b. Results (measured)

| Metric | Baseline | W18–W21 + W22 as first written | **After W22 revert** |
|---|---|---|---|
| **Combined accuracy** | 70.35% | 74.68% | **76.73%** |
| Oracle ceiling (perfect routing) | 74.15% | 81.68% | **81.68%** |
| Gate routing accuracy | 94.41% | 90.79% | 93.40% |
| Gate validation accuracy | 95.31% | 92.76% | 94.46% |
| Best gate-free routing | 75.06% | 77.55% | 77.55% |
| Training time | 137.6s | 200.6s | — |

**Net: +6.38 points combined, +7.53 points on the ceiling.**

### What worked

W18/W18e/W19/W20 moved the **oracle ceiling 74.15% → 81.68%**, i.e. the specialists
themselves got substantially better — the system now beats, with imperfect routing, what
perfect routing could have achieved before. Task-recency bias largely resolved: shard 1's
recall spread across slice order narrowed from 0.23 to 0.10 (airplane 0.61 → 0.79), and
horse stopped over-claiming (recall 0.91 → 0.71 while precision rose 0.55 → 0.77).

### What backfired, and the correction

**W22 cost 2.05 points and was reverted.** Two changes, both wrong:

- **Class weights flipped the bias instead of centring it** — routing went 3679/6321 to
  4521/5479 against a true 4000/6000; shard 1 in-shard retention rose 89% → 95% while
  shard 2 fell 98% → 88%. Because shard 2 holds 6000 of 10000 test samples, the
  unweighted gate's apparent bias toward it was *aggregate-optimal*. Making routing
  per-shard fair made it overall worse. `GATING_CLASS_WEIGHTS = False`.
- **The 19× GAP shrink was not free** — gate validation fell 95.31% → 92.76%. At the
  post-W18 ceiling each routing point is worth ~0.82 system points, so the shrink traded
  ~2.4 points of accuracy for 516k parameters in a model that runs once per sample.
  `GATING_POOL_SIZE = 8`.

Gate augmentation was also split out (`GATING_CROP_PADDING = 0`, `GATING_FLIP_PROB = 0.5`):
routing is a coarse whole-image decision, so a random crop can remove the very content that
separates the shards, unlike the specialists where it is a strong regularizer. Flip stays,
now applied per sample.

Routing recovered to 93.40%, ~1 point below the original 94.41% — attributable to the
per-sample flip changing the augmentation stream, and the gate early-stopping sooner
(15s vs 32s).

### W21 verdict: keep the head, keep the gate

The cosine head sharpened gate-free routing substantially — `head_cosine` 58.62% → 75.00%
— confirming the mechanism. But at 75.00% it remains **18.4 points below the gate**, so it
cannot replace it. `ROUTING_MODE` stays `'gating'`. The cosine head is retained for its
recency-bias benefit, which is real and is part of the ceiling gain above.

Two side effects worth noting: `mahalanobis` collapsed (71.59% → 42.19%) because normalized
features live on a hypersphere where the tied-covariance assumption breaks, and
`energy_debiased` overtook cosine as the best gate-free score (77.55%).

### Next lever

`bird` and `cat` — the earliest classes in shard 2 — are now *under*-predicted
(precision 0.76/0.76, recall 0.58/0.44); the pendulum swung. Likely cause is
`MAX_REPLAY_SAMPLES_PER_CLASS = 1000`: old classes are represented by 1000 of their 4500
images, seen repeatedly, so they overfit those specific samples. That cap did not bind when
replay held 20 of 64 batch slots; at 46 of 64 it does. Raising it (2000–3000, or uncapped)
is the next experiment and needs a full retrain.

## 6. How to run (manual)

`entry_training.py` **skips any shard whose final model already exists**, so the old models must be
cleared or the run will silently load them and report the same numbers. Back them up first if the
70.35% baseline is worth keeping.

```bash
mv "projects/cifar10_sisa_pytorch/models" "projects/cifar10_sisa_pytorch/models_baseline_7035"
python training/entry_training.py
python experiments/gate_free_routing_probe.py       # re-measure routing, no retraining needed
```

Data processing does **not** need re-running — sharding and slicing are unchanged.

Expect training to take noticeably longer than the previous 137s: the balanced replay ratio means
~3.5× more optimizer steps per epoch (300 vs 85 on shard 2's slice 5), which is the stride bug being
fixed rather than a regression.

### Baselines to beat

| Metric | Baseline |
|---|---|
| Combined accuracy | 70.35% |
| Oracle ceiling (perfect routing) | 74.15% |
| Gate routing accuracy | 94.41% |
| Best gate-free routing | 75.06% (`ncm_cosine`) |
| `head_cosine` on linear heads | 58.62% routing — the number W21 should move |

### Reading the result

- **Combined accuracy up, oracle ceiling up** → W18/W19/W20 worked; forgetting was the bottleneck.
- **`head_cosine` routing approaches 94%** → W21 worked; set `ROUTING_MODE='cosine'` and the gating
  network (and W16's per-deletion gate retrain) can be dropped entirely.
- **`head_cosine` stays well below 94%** → keep `ROUTING_MODE='gating'`; W22's class weights and the
  19× smaller gate still apply, and the gate stays as the router.

Switching routing is a config-only change (`config.ROUTING_MODE`) and needs no retraining, so both
can be compared against the same checkpoints.
