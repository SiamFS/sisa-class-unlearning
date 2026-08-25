# SISA Exact Class Unlearning — Master Implementation Plan

**Repository:** `sisa-class-unlearning` · **Paper:** *Machine Unlearning for Class Removal through SISA-based Deep Neural Network Architectures* (arXiv:2604.27804)
**Purpose of this document:** a single, self-contained specification that an AI coding session (with no prior chat context) can execute to fix the codebase and make the work publishable as **efficient, EXACT, provable** SISA class unlearning. Combines three prior review documents and all follow-up design decisions.

> **How to use this file (read first).** Work top to bottom. Section 2 gives you the context you need about the existing code. Section 3 lists locked design decisions and **two decisions the human must confirm before coding** (marked ⚠️ DECIDE). Section 6 is the actual work, split into numbered workstreams W1–W14, each with *files/functions to touch*, *the change*, and *acceptance criteria*. Do not touch the sharding or slicing partitioning logic — see 3.1. Anchor edits by **function name** (line numbers are approximate). After implementing, use Section 12 (Definition of Done) and Section 13 (self-audit) to verify nothing was missed.

---

## 1. Goal and non-negotiables

**Goal:** SISA-based class unlearning that is (a) **exact** — the unlearned model is statistically identical to a model retrained from scratch without the deleted class (SISA Definition III.1); (b) **efficient** — deleting a class retrains only the affected shard (and, if slicing applies, only from the affected slice forward); (c) **provable** — supported by a formula, an empirical test against a from-scratch reference, and visualizations.

**Non-negotiables:**
- Keep the sharding + slicing + **class-sequential** slice ordering. Do NOT change the data-partitioning logic (Section 3.1 explains why).
- Everything must be **reproducible** (global seeding) and **dataset/architecture agnostic** (no hard-coded CIFAR-10 / 10 classes / 32×32).
- Never tune on the test set. Never use a threshold/mask that is adjusted based on the unlearning outcome (that is masking, not forgetting — it breaks the exact claim).

---

## 2. System overview (how the current code works)

**Pipeline:** `data_processing/` → `training/` → `unlearning/`, configured by `config.py`, visualized by `plots.py`, explored by `search.py`.

**Data (`data_processing/entry_data_processing.py`):**
- Loads CIFAR-10, **merges the official train+test into 60k, then re-splits 70/10/20** (stratified, seed 42). ← problem (W2).
- Computes normalization stats from the train split; stores them and class names in `sisa_data/metadata.json`.
- **Sharding (`data_processing/sharding.py`, `create_shards_with_indices`):** class-isolated — each whole class assigned to exactly one shard, load-balanced. `_apply_asymmetric_splitting` is a no-op stub.
- **Slicing (`data_processing/slicing.py`, `create_slices`):** class-sequential — slices filled class-by-class (equal size, class order). A class occupies a contiguous slice block within its shard.
- Saves `shards/shard_{i}/slice_{j}_{x,y,idx}.npy`, `validation_data/`, `test_data/`, and per-shard `metadata.json` with `class_indices_present`.

**Model (`training/create_model.py`):**
- `SISACIFAR10Net(num_classes=10)`: 3 conv blocks (32→64→128) + FC head. **`num_classes=10` is fixed**; `config.FC_LAYER_1_INPUT = 2048 = 128×4×4` assumes 32×32 input. ← agnostic problem (W12).
- `GatingNetwork(num_shards)`: 2 conv + 2 FC, outputs a shard index. `fc1 = Linear(64*8*8, 128)` also assumes 32×32.

**Training (`training/entry_training.py`):**
- Trains the gating network first (`train_gating` in `training/train_gating_model.py`), then each shard **slice-by-slice incrementally**, carrying the model forward and checkpointing after each slice.
- Replay buffer to fight catastrophic forgetting: `training/smart_replay.py` (`SmartReplayBuffer`, gradient-importance + temporal decay) or a simple dict; ratio `config.REPLAY_RATIO = 0.3`.
- `train_model` (`training/train_model.py`) reads several knobs from `config` internally (`WEIGHT_DECAY`, `LABEL_SMOOTHING`, dropout via model creation).
- Final evaluation routes each test sample through the gating net to one shard (`_run_sisa_batch` in `plots.py`).

**Unlearning (`unlearning/sisa_unlearning.py`, class `SISAUnlearning`):**
- `unlearn_by_class(class_name)` / `batch_unlearn_by_classes`: remove class from all slices (`_remove_data_from_slice`), delete emptied slices, retrain the affected shard from the checkpoint **before** the first affected slice (`_retrain_shard_incrementally`), scrub the class from the replay buffer (`_filter_replay_buffer_for_unlearning`), track it in metadata, evaluate.
- **The gating net is NOT retrained** — a confidence/margin threshold at inference (`_run_sisa_batch`, `threshold`) rejects low-confidence routes (→ prediction `-1`). ← exactness problem (W5).
- Metrics for the paper are **scraped from `training.txt`** (`_get_pre_unlearning_class_accuracy`, `_get_training_metrics`, `_parse_classification_report_from_training_log`). ← problem (W7).

**Inference core (`plots.py`, `_run_sisa_batch`):** gating softmax → argmax shard → run that one specialist → temperature scale → optional threshold → `-1`.

---

## 3. Design decisions

### 3.1 LOCKED: keep sharding + slicing + class-sequential ordering
Rationale (do not revisit during implementation):
- **Class-sequential slicing is the only distribution that makes slicing useful for class deletion.** With balanced/random slicing a class is spread across all slices, so deletion touches slice 0 and you retrain the whole shard — slicing buys nothing. Class-sequential concentrates a class into a contiguous slice block, so deletion retrains only from that block forward.
- **Slicing only adds value with multi-class shards** (e.g., 5 classes/shard). This multi-class-shard + class-sequential-slice combination is the project's genuine delta over ARCANE (which is one-class-per-submodel, no slicing).
- Equal-size slices are fine; it is the *ordering* that must stay class-sequential.
- **Optional enhancement (keep architecture):** order classes within a shard by deletion-likelihood so likely-deleted classes sit in later slices → larger expected retrain savings (SISA/ARCANE "sort by erasure probability").

### 3.2 SUPERSEDED BY MEASUREMENT: routing must not rely on a component trained on the deleted data
The current learned gating network is trained on **all** classes including the deleted one and is never retrained → it keeps information about the deleted class → the *system* is not exact even though the specialist retrain is. Resolution priority (both options recorded; option 2 is what's actually in effect — see W16):
1. **Originally preferred — deterministic self-routing (no learned router).** Give each specialist an "is-this-mine" signal and route by it. Implement as: mask each specialist to the classes its shard owns (from `class_indices_present` metadata — not learned) and pick the confident owner; optionally add an OOD/energy/max-softmax score for robustness. Parameter-free routing ⇒ nothing extra to unlearn ⇒ exact by construction. Built and measured (W3, W15): a real 30-trial Optuna study found this caps combined accuracy at 45-63% regardless of hyperparameters, well below either shard's own ~75-90% in-distribution accuracy — an energy-based routing score (W15) was tried and didn't close the gap either. **Rejected in practice, kept as the automatic fallback** (`gating_model=None` in `plots.py::_run_sisa_batch`) for any caller not wired up to a gate.
2. **Now in effect — retrain the lightweight gating net on remaining classes each deletion** (`train_gating(excluded_classes=...)`, wired into `unlearning/sisa_unlearning.py::final_evaluation_with_self_routing` — W16). Measured 58.40%→70.16% combined accuracy (88.08% correct shard-routing) over option 1, both at a smoke scale and the real full training budget. Still efficient (the gate is tiny — ~30s to retrain — so you still avoid retraining the K−1 heavy specialists). This mirrors RecEraser and is exact **only because the gate is actually retrained excluding every unlearned class each time**, not because its output head is coarse (shard-level, not class-level) — a network's weights reflect gradient updates from whatever data it saw regardless of what the output layer represents, so an un-retrained gate would still carry a detectable fingerprint of the deleted class's images (would show up as a membership-inference signal).
3. The confidence/margin **threshold is removed from the exactness claim** (see W5). If kept at all, it is a deployment-time rejection feature, tuned once on validation, reported separately, never as unlearning evidence.

### 3.3 DECIDED: Lane A only (frozen-backbone Lane B rejected)
Class-sequential slicing = class-incremental training = catastrophic forgetting (root cause of the ~73% accuracy). Two lanes were considered, both keeping sharding+slicing:
- **Lane A — from-scratch CNN + simple, seeded replay (W4).** Lowest effort, keeps everything, clean exact proof.
- **Lane B — frozen ImageNet-pretrained backbone + per-slice/class LoRA-style adapters.** Was prototyped in-session, then rejected. Two reasons, both load-bearing:
  1. **Exactness gets a real caveat, not just a nuance.** A frozen backbone never trains on this project's data, so it satisfies the letter of "never trained on the deleted class's data" — but it *does* already encode general visual knowledge from ImageNet (e.g. it already "knows" what a cat looks like before the pipeline ever runs). The unlearned-vs-scratch comparison (§4.1) still holds, because both sides use the identical frozen backbone — but the system can never truthfully claim "this model has no ability to recognize class *c*," only "no project-specific, trainable component for class *c* remains." Lane A carries no such caveat: nothing in it has ever seen anything except this project's own data.
  2. **Its efficiency promise conflicts with replay.** The whole point of per-slice adapters is that each one is trained *only* on its own slice, so deleting it removes exactly and only that slice's contribution, with zero retraining. But replay (needed for accuracy — see W4) works by deliberately mixing earlier slices' samples into later slices' training. Applying replay to an adapter breaks its isolation (it now encodes other slices' classes too); withholding replay to preserve isolation reintroduces the catastrophic-forgetting problem replay exists to solve. The two don't compose.

**Decision: Lane A only.** All Lane B code (frozen backbone, per-slice adapters, `--architecture` flag, `SmartReplayBuffer`) was removed. `SmartReplayBuffer` specifically was also unseeded (`np.random.choice` on the global RNG, not a seeded generator) — non-deterministic, so it could never be part of the exactness claim either way; confirmed it still ran without crashing before deleting it.

### 3.4 DECIDED: dynamic head sizing after unlearning (option b)
Implemented. `unlearning/sisa_unlearning.py::_resize_model_head` retrains the affected shard with a head sized to its current (post-unlearning) owned-class count: the convolutional trunk and hidden FC layer are transplanted from the last clean checkpoint, only the final classification layer is freshly initialized at the new width. `training/train_model.py`'s `head_classes` parameter drives the global↔local label remap for training and validation; `plots.py::_scatter_local_to_global` (used by `_run_sisa_batch`, `create_shard_confusion_matrix`, and `train_model`'s validation) places a reduced-width specialist's output back into global class space so routing/evaluation code needed no other changes. The deleted class ends up with no output neuron at all, not a masked one.

### 3.5 LOCKED: reproducible + agnostic + Optuna-once
Global seeding (W1); dataset/architecture agnostic (W12); one-time Optuna study on validation, freeze into config, retire the tool to `tuning/` (W13).

---

## 4. Making it EXACT + the proof (formula / test / visualization)

### 4.1 Exactness definition (state this in the paper)
```
Distribution( Unlearn( Train(D ∪ Dc), Dc ) )  ==  Distribution( Train(D \ Dc) )
```
"Exact" ⇒ the unlearned model equals (in distribution) a model that never saw class c. Accuracy ≈ 0 on the deleted class is necessary but NOT sufficient.

### 4.2 The four conditions (must all hold)
1. Retrain the affected shard from the last **class-clean** checkpoint — **already implemented** (`_retrain_shard_incrementally` loads the model from `first_affected_slice-1`; class-sequential ⇒ that checkpoint never saw the class). Verify with W14/W8.
2. Scrub the deleted class from replay everywhere — **implemented** (`_filter_replay_buffer_for_unlearning`), but make replay deterministic (W4).
3. **Determinism** (W1) — required so the unlearned model can equal the scratch model.
4. **Unlearn the router** — resolved by self-routing (3.2) so there is nothing to unlearn, or by retraining the gate (fallback).

### 4.3 Formula (trajectory equivalence — put in the paper)
Let the affected shard's slices be s₁…s_L and let class c first appear at slice j. With seeded deterministic training θ:
- `θ_{j-1}` is a function of s₁…s_{j-1}, none containing c ⇒ identical to the scratch model at that point.
- Retraining s_j…s_L with c removed from data and replay, starting from `θ_{j-1}`, applies the same deterministic map to the same inputs as a from-scratch run without c ⇒ identical specialist.
- Self-routing is parameter-free (or the gate is retrained on D\Dc) ⇒ identical routing.
⇒ system equals the from-scratch-without-c system (Def. III.1). Report SISA speedup up to `(R+1)·S/2×`.

### 4.4 Test (implement in W8)
Build a **from-scratch reference**: train the whole system on the dataset with class c removed from the start, using the **identical frozen config, seed, slice schedule, and head sizing**. Compare unlearned vs scratch:
1. **Parameter distance** (L2 / cosine) per layer — ≈ 0 with a shared seed; small & stable across seeds otherwise.
2. **Prediction agreement** on the test set — ≈ 100%.
3. **Output-distribution distance** — mean KL / total-variation between softmax outputs.
4. **Membership-inference AUC on the deleted class** — ≈ 0.5 (indistinguishable from never-trained) and equal to the scratch model's. This is the privacy proof.
5. **Deleted-class accuracy vs the scratch reference** — equal to what scratch scores on c (not "≈0 in isolation").

### 4.5 Visualization (W8)
- Parameter-distance and prediction-agreement bars: original vs unlearned vs scratch.
- MIA ROC curves with AUC annotated (original high on c; unlearned ≈ 0.5 = scratch).
- Confusion matrices before / after / scratch, side by side.
- Efficiency: retraining time vs full retrain across (shards × slices) with the theoretical speedup overlaid.

---

## 5. Novelty positioning (required — reviewers will check)
Two precedents overlap heavily and MUST be cited and compared:
- **ARCANE (IJCAI 2022):** class-wise partition, one sub-model per class, exact unlearning, benchmarks vs SISA on MNIST/CIFAR-10/ImageNet. → your class-isolated sharding.
- **RecEraser (WWW 2022):** replaces SISA's static vote with a **learned aggregator that is retrained on unlearning**. → your gating router idea; the "retrain the router" fallback is their contribution.

**Your defensible delta:** multi-class shards + **class-sequential slicing** (finer retrain granularity than ARCANE's whole-submodel retrain) + **parameter-free self-routing** (exact by construction, unlike RecEraser's retrained aggregator) + a rigorous **exactness proof with MIA**. To be competitive at a selective venue, add: comparison to ARCANE/RecEraser and modern class-unlearning baselines (SSD, UNSIR, Boundary Unlearning), and ideally scale beyond CIFAR-10 (CIFAR-100 / Tiny-ImageNet), enabled by the agnostic refactor (W12). Without a new idea + baselines this is a workshop/thesis/arXiv-level contribution even after all fixes.

---

## 6. Implementation workstreams

> Each workstream: **Files/anchors → Change → Acceptance criteria.** Suggested order in Section 9.

### W1 — Global determinism (foundation; do first)
- **Files:** new `utils/seeding.py`; call at the top of every entry point (`data_processing/entry_data_processing.py`, `training/entry_training.py`, `unlearning/sisa_unlearning.py` main, `tune.py`, and the new scratch-reference script).
- **Change:** `set_seed(seed)` seeding `random`, `numpy`, `torch`, `torch.cuda`; set `cudnn.deterministic=True`, `cudnn.benchmark=False`, `torch.use_deterministic_algorithms(True, warn_only=True)`, `PYTHONHASHSEED`. Give every `DataLoader` and the replay sampler a seeded `torch.Generator` / `np.random.default_rng(seed)`. Persist the seed in each run's output.
- **Acceptance:** two consecutive full training runs with the same seed produce identical final test accuracy (and identical weights on CPU).

### W2 — Canonical data split
- **Files:** `data_processing/entry_data_processing.py` (`load_cifar10_data`).
- **Change:** stop merging train+test. Use the official 50k train (carve validation from it, e.g. 45k/5k stratified) and keep the official **10k test set untouched**. Recompute normalization from the train split only. Keep writing `metadata.json`.
- **Acceptance:** `test_data/` equals the standard CIFAR-10 test set (10k, 1k/class); no training/val sample appears in test; normalization computed on train only.

### W3 — Specialist head + self-routing (accuracy + exact routing)
- **Files:** `training/create_model.py` (`SISACIFAR10Net`), `plots.py` (`_run_sisa_batch`), `unlearning/sisa_unlearning.py` (evaluation), `search.py` (`_predict_with_true_sisa_batch`).
- **Change:**
  - Replace learned-gating routing in `_run_sisa_batch` with **deterministic self-routing**: run each specialist, mask its logits to `class_indices_present` for its shard (read from shard metadata — pass the shard→owned-classes map in), take the max-confidence owned class across shards. Optionally add an OOD/energy score for tie-breaking.
  - Keep a single implementation used by training eval, unlearning eval, and `search.py` (one source of truth).
- **Acceptance:** with self-routing (no learned gate), system accuracy on validation is ≥ the learned-gate accuracy; deleting a class needs no router change; no code path requires the gating network for correctness.
- **Note:** if accuracy is insufficient, apply 3.2 fallback (retrain gate with `excluded_classes`).

### W4 — Replay: smart → simple, deterministic, capped [DONE]
- **Files:** `training/replay_buffer.py` (new: `add_to_replay_buffer`), `training/train_model.py`, `training/entry_training.py`, `unlearning/sisa_unlearning.py` (`_retrain_shard_incrementally`), `config.py`. `training/smart_replay.py` deleted.
- **Change:** the simple per-class dict replay is now the only mechanism (`SmartReplayBuffer` removed entirely — see 3.3), sampled with a **seeded** `np.random.default_rng(config.SEED)` (was previously hardcoded to a stray literal `42`, a real reproducibility bug, now fixed). Additionally capped: `MAX_REPLAY_SAMPLES_PER_CLASS` (now applied to the simple buffer, not just the old smart one) evicts down to the cap via seeded random subsampling whenever a class would exceed it. This was a real bug found by audit — the buffer accumulates data from every prior slice (and, during retrain, every already-redone slice too), and had no cap at all, so memory and buffer-build time grew unboundedly across a shard's slices / repeated unlearning rounds. Per-batch training cost was never affected (`replay_ratio` samples a fixed proportion per batch regardless of buffer size), only memory and buffer-construction time were.
- **Acceptance:** replay sampling is reproducible; a class's buffer entry never exceeds `MAX_REPLAY_SAMPLES_PER_CLASS`; unlearning retrain from a clean checkpoint reproduces the scratch specialist (feeds W8).

### W5 — Router unlearning + drop threshold from the claim [DONE]
- **Files:** `unlearning/sisa_unlearning.py` (`unlearn_by_class`, `evaluate_with_self_routing`).
- **Change:** removed the stale "Skipping gating network retraining… confidence threshold" print (`unlearn_by_class`, Step 3) — there's no router to unlearn under self-routing, so there was never anything being skipped. `evaluate_with_self_routing`'s `threshold` default changed from `config.CONFIDENCE_THRESHOLD` to `None`; its sole caller (`final_evaluation_with_self_routing`, used for every unlearning report) now gets raw, no-threshold numbers by default. The thresholded path still exists but only runs if a caller explicitly passes a threshold, and is now labeled `"DEPLOYMENT-ONLY... (NOT exactness evidence)"` in its own output — isolated and clearly non-evidence, per the plan.
- **Acceptance:** verified end-to-end against a real trained project (copied, unlearned, discarded) — primary evaluation output reads `"Self-Routing Evaluation (raw, no confidence threshold)"`, no crash, no stale message, deployment-only path never triggers unless explicitly requested.

### W6 — Head/class-count after unlearning (per decision 3.4) [DONE]
- **Files:** `training/create_model.py` (`_resize_model_head` logic lives in `unlearning/sisa_unlearning.py`; `create_model.py`'s loader infers a checkpoint's own head width from its state dict rather than assuming the global count), `training/train_model.py` (`head_classes` param), `plots.py` (`_scatter_local_to_global`, used by `_run_sisa_batch` and `create_shard_confusion_matrix`), `unlearning/sisa_unlearning.py` (`_resize_model_head`, `_retrain_shard_incrementally`).
- **Change (option b, implemented):** `_retrain_shard_incrementally` retrains the affected shard with a head sized to its current owned-class count (`_resize_model_head`: transplants conv + hidden-FC weights from the last clean checkpoint, reinitializes only the final Linear layer at the new width). Training/validation labels are remapped global↔local via `head_classes`; a reduced-width specialist's output is scattered back to global class space wherever inference needs a global-space vector, so `_run_sisa_batch`/self-routing/confusion-matrix code needed no further changes and transparently handles both global-width (never-unlearned) and reduced-width (post-unlearning) shards side by side.
- **Note:** the from-scratch scratch-reference harness (W8, not yet built) must reuse this same `head_classes` mechanism to match sizing.
- **Acceptance:** the unlearned model's final layer literally has no output neuron for the deleted class (verified: `model.fc_layer[-1].out_features == len(class_indices_present)`); self-routing/prediction correctness for other classes is unaffected (smoke-tested with mixed global/reduced-width shards in the same system).

### W7 — Metrics computed directly (kill the log-scraping) [DONE]
- **Files:** `training/entry_training.py` (`evaluate_with_self_routing` now also returns a `classification_report(..., output_dict=True)`; `__main__` assembles and `json.dump`s `sisa_data/training_metrics.json`), `unlearning/sisa_unlearning.py` (`_get_training_metrics`, `_get_pre_unlearning_class_accuracy`; `_parse_classification_report_from_training_log` deleted entirely).
- **Change:** `_get_training_metrics`/`_get_pre_unlearning_class_accuracy` now `json.load(training_metrics.json)` instead of regex-scraping a relative, non-project-scoped `training.txt` (a real bug: with more than one project, that path could silently read the wrong project's numbers). The hardcoded `0.74` fallback is gone — a missing/incomplete metrics file now raises clearly instead of returning a guessed number.
- **Acceptance:** verified end-to-end on a throwaway temp project (data processing → training → unlearning via the real CLI) — `training_metrics.json` has all expected keys with a real classification report, unlearning's pre-/post-unlearning accuracy reporting reads it correctly with no crash, and deleting `training.txt` (which is no longer even written) doesn't change any reported metric.

### W8 — Exactness proof harness (the core new deliverable) [DONE]
- **Files:** `experiments/scratch_reference.py` (new), `experiments/exactness_eval.py` (new), 4 new plot functions in `plots.py` (`create_exactness_comparison_chart`, `create_mia_roc_chart`, `create_exactness_confusion_matrices`, `create_efficiency_comparison_chart`).
- **Scope decisions locked in** (worked out through discussion, not guessed): compares the **affected shard only** (unaffected shards are provably untouched by W6, nothing to prove there); the scratch reference is a **true, independent full retrain from slice 1** (not a checkpoint-reuse shortcut — a shortcut would just be comparing real unlearning's own retrain code against itself, unable to catch a bug that lives in that shared code; confirmed as the field's "gold standard" and matches this project's own source paper's Model-1 baseline methodology); MIA is **confidence-based**, not shadow-model.
- **Change:** `scratch_reference.py::build_scratch_reference` — works from copies only (pristine source project never modified): snapshots the class's real training samples first (MIA "member" set, since real unlearning would otherwise destroy them), copies the project, reuses `SISAUnlearning._remove_data_from_slice`/`_update_shard_metadata` (not reimplemented) to strip the class from the affected shard's slices, then trains that one shard from a fresh random model through every slice with a head fixed from the start to the shard's final owned classes (W6-style) and validation filtered to exclude the deleted class throughout. `exactness_eval.py::run_exactness_eval` — the entry point: runs real (unmodified) `unlearn_by_class` on its own separate copy, builds the scratch reference, then computes parameter distance, prediction agreement, output-distribution KL, MIA AUC (original/unlearned/scratch), and deleted-class prediction agreement, comparing full-system self-routing with the affected shard swapped between the unlearned and scratch models (a headless dynamic-head shard can't score "accuracy" on a class it has no output neuron for, so the comparison has to happen at the full-system level). MIA uses max-softmax-confidence as the membership signal, not loss-against-true-label, because the deleted class's own logit doesn't exist on either model. Efficiency numbers come for free from the scratch run's own measured training time vs. real unlearning's already-tracked `pure_retraining_time`.
- **Acceptance:** verified end-to-end (data processing → training → real unlearning → scratch reference → full comparison) on a throwaway project — all 5 metrics + MIA AUCs computed and in valid range, all 4 plots + the JSON report produced, pristine source project confirmed untouched throughout.
- **Documented, not fixed:** parameter distance is not expected to be ~0 — the scratch script's RNG consumption sequence differs from the original run's (seed→straight into one shard, vs. seed→gating net→other shards→this shard), so this is the exact "GPU non-determinism" fallback plan §7 already anticipates. The report and plots state this explicitly and treat the behavioral metrics (prediction agreement, output distribution, MIA) as the primary evidence.

### W9 — Visualization bug fixes [DONE]
- **Files:** `plots.py`, `training/entry_training.py`, `unlearning/sisa_unlearning.py`.
- **Changes:**
  - `create_overall_dataset_visualization`: training total used `len(slice[0]['y'])*num_slices` (assumed equal slices) — replaced with `sum(len(s['y']) for shard in shards_data for s in shard['slices'])`.
  - Removed the duplicate defs of `_normalize_probabilities_tensor`/`_apply_temperature_tensor` (were defined twice — kept one).
  - `create_overall_sisa_training_curves`: removed the `combined_*`/`max_epochs`/`shard_epoch_counts` aggregation — computed but never read or plotted anywhere.
  - De-duplicated redundant full-test inference: `create_overall_sisa_roc_curve`/`create_overall_sisa_confusion_matrix` now accept an optional `precomputed=(preds, probs, labels)`, skipping their own `_run_sisa_batch` pass when given one. `training/entry_training.py`'s `evaluate_with_self_routing` now returns its own pass for reuse (3 passes → 1 for the final-evaluation section). `unlearning/sisa_unlearning.py::evaluate_with_self_routing`'s "active classes" re-evaluation now captures probs too and threads that single pass through all 6 downstream ROC/confusion-matrix calls (8 passes → 2: the threshold-aware primary pass, kept separate since its semantics differ, plus one shared raw pass).
- **Acceptance:** verified end-to-end (fresh training run + real single-class unlearning) — all expected plot files still produced correctly through the deduplicated paths, no crashes, dataset split proportions correct.

### W10 — Logging via stdlib [DONE]
- **Files:** new `utils/run_logging.py` (`setup_run_logging`); `data_processing/entry_data_processing.py`, `training/entry_training.py`, `unlearning/sisa_unlearning.py` (the old `Logger`/`TrainingLogger`/`UnlearningLogger` classes deleted).
- **Change:** the hundreds of existing `print()` calls were deliberately left untouched (migrating them all to `logging.info()` was judged too large a blast radius for no acceptance-criteria benefit). Instead, a stdlib `logging.FileHandler` sits behind a thin `sys.stdout` redirect (`_TeeToLogger`) -- every `print()` keeps working unchanged, but the file is opened/managed by `logging`, not a hand-rolled `open(path, 'w')` that truncated on every run. Each entry point calls `setup_run_logging(os.path.join(base_dir, "logs"), "<name>")` once `base_dir` (or, for `unlearning/sisa_unlearning.py`, `--project-name`) is known -- all three had their logging setup moved from before that point to after it, since it was previously initialized before the project name was even known.
- **Acceptance:** logs land at `projects/<name>/logs/{data_processing,training,unlearning}_<timestamp>.log` -- project-scoped and timestamped. Verified two consecutive training runs against the same project produce two distinct, non-overwritten log files (the actual bug). No code parses log text for numbers (W7). Running from any directory already worked before this pass (both old Logger classes anchored to `config.PROJECT_ROOT`) and still does.

### W11 — Remove dead / submission-noise code [DONE]
- **Files:** `config.py`, `data_processing/sharding.py`, `unlearning/sisa_unlearning.py`.
- **Change:** `search.py`'s `_predict_with_gating` was already gone (confirmed via grep, nothing to remove). Removed the decorative, never-referenced `ALPHA` constant from `sharding.py` (`BETA`/`GAMMA`/`MAX_IMBALANCE_RATIO` are all actually used, kept; `_apply_asymmetric_splitting` is called, not dead, left as-is per §7's own guidance that real asymmetric splitting is out of scope). Removed dead config params confirmed to have zero references anywhere in the codebase: `USE_REPLAY_BUFFER`, `GATING_MARGIN_THRESHOLD`, `PRIMARY_SPECIALIST_WEIGHT`, `CONFIDENCE_BOOST_THRESHOLD` (all leftovers from the pre-self-routing learned-gating design). Removed `self.gating_update_time` (set three times across `unlearn_by_class`/`batch_unlearn_by_classes`, read nowhere).
- **Acceptance:** grep confirms no remaining references to any removed symbol; verified end-to-end (data processing + training + unlearning) that none of the removals broke anything.

### W12 — Dataset / architecture agnostic [PARTIAL -- factory built, second dataset not verified]
- **Files:** `data_processing/datasets.py` (new: `DATASET_REGISTRY`/`get_dataset_class`), `data_processing/entry_data_processing.py`, `training/create_model.py` (`SISACIFAR10Net` renamed → `SISAConvNet`), `config.py` (`DATASET = "cifar10"`, machine-readable key distinct from the `DATASET_NAME` display string).
- **Change:** `entry_data_processing.py` now resolves `config.DATASET` through the factory instead of hardcoding `torchvision.datasets.CIFAR10`; `load_cifar10_data()` renamed `load_dataset()`. `num_classes`/class names were already derived from `metadata.json` everywhere (confirmed no hardcoded `10` or CIFAR class-name lists remain — `search.py` already raises rather than falling back to a hardcoded list; the old log-scraping class list was removed entirely by W7). Model class renamed to the dataset-agnostic `SISAConvNet`; verified old checkpoints saved under the previous class name still load fine (the loader only pattern-matches `'GatingNetwork'`, nothing else). CIFAR-100 is pre-registered in the factory (same 32×32×3 shape, has `.classes`) and should work by changing `config.DATASET` alone.
- **Scope decision (per discussion):** did not download/run a second real dataset through the full pipeline to prove the acceptance criterion literally — that's a real download + real training run, deferred until actually needed. Everything that's normally a code-level blocker (hardcoded loader, hardcoded class names, hardcoded model name) is fixed and was verified via a real, fast end-to-end run (data processing through training) with the new factory/renamed model on CIFAR-10.
- **Acceptance:** no literal `CIFAR`/`10` dataset assumptions remain in logic (grep-verified); switching `config.DATASET` to a registered second dataset is a config-only change, not yet proven by an actual second-dataset run.
- **Still open (not done, out of scope for this pass):** no separate `build_model(name, ...)` architecture factory (only the CNN exists) — deferred, no concrete need yet since only one architecture is in use.

### W13 — Optuna one-time, freeze, retire [DONE]
- **Files:** `tuning/tune.py` (new — no `tune.py` existed anywhere in the repo despite the plan's "provided" note; written from scratch).
- **Change:** seeded `TPESampler`, searches `LEARNING_RATE`/`WEIGHT_DECAY`/`LABEL_SMOOTHING`/`REPLAY_RATIO`/`BATCH_SIZE`/`FC_LAYER_DROPOUT` on **validation only**, using Lane A's simple deterministic replay (never `SmartReplayBuffer`, which no longer exists). Each trial runs the **entire SISA pipeline** — both shards, all 5 slices, real `config.MAX_EPOCHS` budget with early stopping, balance-based augmentation — scored by full-system self-routing accuracy on validation data (gating network intentionally skipped per trial: it's diagnostic-only and never affects self-routing predictions or the score being optimized). Logs via the same `setup_run_logging` (W10) every other entry point uses, writing to `projects/<name>/logs/tuning_<timestamp>.log` rather than a loose root-level file (an earlier version redirected stdout to a bare repo-root file via shell `>`, which got deleted out from under the running process mid-study on Windows — orphaning the handle and losing that run's log; moving it into the project's own logs directory avoids that class of accident). Writes `best_config.json` with the winning params + provenance (seed, project, trial count, epoch budget, timestamp).
- **Real study run:** 30 trials, full pipeline, real epoch budget, completed. Best validation accuracy **0.6266** (`LEARNING_RATE≈0.000849`, `WEIGHT_DECAY≈3.59e-5`, `LABEL_SMOOTHING≈0.291`, `REPLAY_RATIO≈0.410`, `BATCH_SIZE=32`, `FC_LAYER_DROPOUT≈0.569`). All 30 trials clustered in 0.45–0.63 regardless of hyperparameters — this hyperparameter-independence is what led directly to the W15/W16 investigation below. Values have **not** been frozen into `config.py` yet — deferred until tuning is updated for the new gating-based routing (W16's own deferred follow-up list), since re-tuning under the new objective would likely shift the winning values and `best_config.json` was measured under the now-superseded routing mechanism.
- **Acceptance:** training/unlearning import nothing from Optuna (confirmed — `tuning/tune.py` is the only file in the repo that imports it).

### W15 — Energy-based cross-shard routing investigation [DONE -- rejected, documented so it isn't re-litigated]
- **Motivation:** the W13 study surfaced a consistent, hyperparameter-independent gap: each shard specialist reaches ~75-90% validation accuracy on its own owned classes (visible per-slice in the tuning logs), but the combined self-routing system only reaches ~45-63% (all 30 trials). Root cause traced to `plots.py::_run_sisa_batch`: the routing decision (which shard's prediction wins) is a raw softmax-confidence comparison across independently-trained specialists — a well-known failure mode (softmax classifiers stay confidently "opinionated" even on inputs from classes they've never seen). The literature's established fix for exactly this ("task-id inference via OOD detection" in class-incremental learning; energy-based OOD scores, Liu et al. NeurIPS 2020, arXiv:2010.03759) computes a routing score from raw logits via `logsumexp` instead of softmax, which theoretically retains more separation information. `IMPLEMENTATION_PLAN.md` §3.2/W3 had already named this as a deferred option ("Optionally add an OOD/energy score for tie-breaking") — this investigation finished that thread.
- **Validation method:** built `experiments/energy_routing_probe.py` (kept in the repo as documentation of this negative result, not deleted) — trains both shards/all slices once via the reusable per-shard/per-slice helpers already in `tuning/tune.py`, then evaluates the SAME trained models with (a) the existing softmax-max routing (baseline) and (b) a candidate energy-based score, `T*(logsumexp(owned_logits/T) - log(k))` (the `log(k)` term de-biases `logsumexp`'s growth with class count — necessary because W6 unlearning shrinks one shard's owned-class count relative to the other, which would otherwise hand the untouched shard a free edge). Tested first at a 5-epoch smoke scale (script correctness only), then at the real full `config.MAX_EPOCHS=100`/`TRAINING_PATIENCE=7` budget (a fair comparison against the same budget the W13 baseline numbers were measured at).
- **Result: no real improvement.** Full-budget run: baseline accuracy 0.5840 vs. energy-based 0.5834 — statistically indistinguishable. Per-shard score separability was present but weak and not consistently comparable *across* shards (e.g. shard 1's out-of-distribution score mean sat close to shard 2's in-distribution mean), which is the literature's own documented caveat for energy scores: raw logit magnitude has no cross-network calibration constraint, so two independently-trained specialists can have systematically different logit scales for reasons unrelated to whether an input belongs to them. Softmax confidence at least bounds each specialist to `[0,1]`; energy score removes that bound without adding calibration back, so in practice it did not out-perform the baseline here.
- **Disposition:** did not modify `plots.py::_run_sisa_batch` or add a `ROUTING_ENERGY_TEMPERATURE` config constant — the gate ("only if the probe shows a real, routing-attributable improvement") was not met. The accuracy ceiling from confidence-based self-routing without a learned gate is a genuine, now-measured cost of that architectural choice (traded off against never having anything to unlearn in the router itself), not a bug in the current implementation. A future attempt at closing this gap would need genuine cross-shard calibration (e.g. per-shard temperature/Platt scaling fit against a shared reference, not just a raw-logit score) rather than a drop-in scoring-function swap.

### W16 — Switch real routing to the gating network, retrained on unlearning [DONE]
- **Motivation:** W15's rejection left the ~45-63% self-routing ceiling unresolved. A follow-up probe tested the ALREADY-BUILT `GatingNetwork` (`training/create_model.py`), trained during `entry_training.py` since before this session but used only for diagnostic plots, never real predictions. Using it as the real router measured **0.5840 → 0.7016** combined accuracy (88.08% correct shard-routing) — a real, large improvement, confirmed at both a 5-epoch smoke scale and the real full `config.MAX_EPOCHS=100` budget (`experiments/gating_routing_probe.py`, kept in the repo). A tree-based router (`RandomForestClassifier`/`GradientBoostingClassifier` on flattened raw pixels) was also tested and confirmed worse (76.33%/73.39%) and slower to train than the CNN gate (87.74%) — trees have no spatial inductive bias for image data, so this wasn't a surprise, but it was verified rather than assumed.
- **Files:** `plots.py` (`_run_sisa_batch`, new `_route_via_gating` helper), `training/entry_training.py` (`evaluate_with_self_routing`, the real post-training call site), `unlearning/sisa_unlearning.py` (`final_evaluation_with_self_routing`, `evaluate_with_self_routing`, `_evaluate_on_forgotten_samples`, `_evaluate_sisa_on_data`), `search.py` (new `_load_gating_model`, `_predict_with_true_sisa_batch`).
- **Change:** `_run_sisa_batch` gained an optional `gating_model=None` parameter. When given, the gate's `argmax` picks the winning shard directly, and that shard's own softmax (computed directly over its gathered owned-logits, not the old full-width masked-then-renormalized approach) is scattered to global width for reporting. When `None` (default), falls through to the original confidence-based routing unchanged — the fallback path for any caller not wired to a gate. `unlearning/sisa_unlearning.py::final_evaluation_with_self_routing` (the single method both `unlearn_by_class` and `batch_unlearn_by_classes` call) now retrains the gate via `train_gating(excluded_classes=self.get_unlearned_classes())` every time, overwriting `models/gating_model.pth`, before loading it and passing it into evaluation — this is the exactness-critical step: a network's weights reflect gradient updates from whatever data it was trained on regardless of what its output layer represents, so an un-retrained gate would carry a detectable fingerprint of a deleted class's images even though its output is only ever "which shard," not "which class." Timed as a new "Gating Retrain Time" line in both unlearning methods' existing timing breakdown printouts.
- **Deferred (flagged, not silently dropped):** `tuning/tune.py` still skips gating training per trial with now-stale reasoning ("diagnostic only") — re-tuning under the new regime needs the tuning script to also train a gate per trial, plus a `save_dir` override on `train_gating` so a hyperparameter search doesn't overwrite the real project's `models/gating_model.pth` as a side effect. `best_config.json` (measured under the old objective) is stale and still not frozen into `config.py`. `experiments/scratch_reference.py`/`exactness_eval.py` (the W8 harness) don't yet train a matching gate for the scratch-reference system, so a from-scratch W8 comparison right now would be apples-to-oranges (real system uses gating routing, scratch system doesn't) — needs updating before the next full exactness run.
- **Acceptance:** compiled clean on every touched file; verified end-to-end on a throwaway copy of the real trained project (training + a real single-class unlearning run), confirmed the gate retrains and its timing prints, confirmed `search.py` runs both with and without a gating model present.

### W17 — Semantic (WordNet) class clustering for sharding [DONE, validated -- NOT yet applied to the real project]
- **Motivation:** W16 found gating-routing accuracy (88.08%), not per-shard classifier quality (~75-90%), was the actual combined-accuracy bottleneck. The existing shard assignment (`data_processing/sharding.py::_find_balanced_shard`) groups classes purely by sample-count balance, with no regard for whether the resulting shards are visually/semantically distinct from each other — the real project's assignment mixed vehicles and animals into both shards (an even/odd class-index artifact, not a deliberate choice). If shards are semantically coherent internally and distinct from each other, the gate's job gets easier by construction.
- **Files:** `data_processing/semantic_clustering.py` (new — `_resolve_synset`, `compute_semantic_similarity_matrix`, `cluster_classes_semantically`), `data_processing/sharding.py::_create_class_isolated_shards` (standard-assignment branch only; the `BETA`/`classes_to_split`/`_apply_asymmetric_splitting` oversized-class path is untouched).
- **Change:** classes are now clustered by WordNet (NLTK) Wu-Palmer semantic similarity over their real name strings — zero training, zero image data touched, computed once during data processing before any shard/slice training exists. Verified directly on CIFAR-10: cleanly finds {airplane, automobile, ship, truck} vs {bird, cat, deer, dog, frog, horse} (within-group similarity 0.67-0.92, cross-group 0.30-0.40) — a natural 4-vs-6 class split, confirmed acceptable (sample-count, not class-count, is what `MAX_IMBALANCE_RATIO` guards, and 6000/4000=1.5x is well within its 3.0 threshold). `automobile` resolves directly to WordNet's `car.n.01` with no manual synonym mapping; a "try full name → underscores-as-spaces → last word → first word" fallback resolves CIFAR-100-style compound names (e.g. `aquarium_fish` → `fish`) with no direct synset of their own, keeping the approach dataset-agnostic per W12. Falls back to the original `_find_balanced_shard` balance algorithm (with a printed warning) if WordNet data can't load, or if `NUM_SHARDS` exceeds the number of classes (clustering can't produce more clusters than samples — confirmed via a direct test that this raises `ValueError` without the guard; the existing empty-shard-skip logic already tolerates this case once the guard routes around clustering).
- **Validated empirically** (throwaway project, never the real one): full data-processing run confirmed the exact predicted shard composition; `experiments/gating_routing_probe.py` (existing, reused as-is) measured **routing accuracy 87.74%→94.56%** and **combined accuracy 70.16%→72.06%** at the real full training budget — a real but more modest combined-accuracy gain than the routing number alone suggests, because grouping all 6 animal classes together makes them easier to route to as a group but somewhat harder to tell apart from each other once there (a genuinely tougher fine-grained 6-way problem than the old, more varied 5-class mix) — reported as measured, not oversold. Separately confirmed `NUM_SHARDS=3`/`NUM_SLICES_PER_SHARD=4` runs cleanly end-to-end (splits the vehicle group further into {airplane, ship} vs {automobile, truck} once a 3rd shard is available) — `NUM_SLICES_PER_SHARD` is fully independent of shard composition since slicing only ever operates on whatever data already landed in a shard.
- **Explicitly out of scope this pass:** auto-deriving `NUM_SHARDS`/`NUM_SLICES_PER_SHARD` from the clustering result (stays a fixed config constant); reusing this clustering's embeddings inside the gating network itself (the clustering step is one-time/offline and never touches the live, retrained-on-unlearning pipeline, so there's no shared-weight concern to design around).
- **Not yet applied to the real project** (`cifar10_sisa_pytorch`): doing so means regenerating its sharded data from scratch (destroys the current shard/slice files) and re-running real training (and likely re-tuning, since shard composition changed) — a separate, deliberate decision, not bundled into this implementation pass.
- **Dependency:** `nltk` installed into both Python environments this project uses (`py -3.8` and plain `python`/3.12). No dependency manifest exists anywhere in this repo (confirmed — not even `optuna`, already imported by `tuning/tune.py`, is tracked anywhere), so this doesn't deviate from existing practice.

### W14 — Edge-case guards (see Section 7 for the full list)
- **Files:** across `training/train_model.py`, `training/early_stopping.py`, `training/smart_replay.py`, `unlearning/sisa_unlearning.py`.
- **Acceptance:** every case in Section 7 is handled or explicitly asserted.

---

## 7. Edge-case catalog (verify each)

**Unlearning logic:**
- Class in slice 0 → no previous checkpoint → retrain shard from scratch (currently: `current_model=None`). ✔ keep; add a test.
- Class spanning multiple slices / sharing a boundary slice with another class → first-affected-slice tracking + per-slice removal. ✔ verify.
- Deleting a class that empties an entire shard → must raise before breaking the system. ✔ present; keep the guard.
- Gaps from deleted slices during retrain → slice-state skip logic. ✔ verify.
- Batch / sequential deletion of multiple classes → prove exactness holds across a sequence, not just one request. Add a test.
- Deleted-class artifacts on disk: stale slice checkpoints from `first_affected_slice` onward are deleted; confirm no checkpoint that saw the class survives. Also: `_save_forgotten_class_samples` and `search.py` **retain deleted-class images** for "verification" — for a privacy claim, gate this behind a debug flag and exclude from any shipped artifact.

**Training / model:**
- `train_model` with `validation_data` provided but `active_classes=None` → crashes at `sorted(active_classes)`. Add a guard.
- `SmartReplayBuffer.sample_for_replay` empty-return shape is `(0,32,32,3)` (HWC) while the pipeline is CHW — return CHW-shaped empties (moot if Lane A uses simple replay).
- `early_stopping.restore_best_model` relies on `hasattr(model,'device')` (always False) → best weights restored to CPU regardless of device. Track the device explicitly.
- Single-shard config → gating margin path in `_run_sisa_batch` degenerates (moot under self-routing; guard anyway).
- Specialist over-confidence on non-owned classes (the accuracy collapse) → fixed by masked self-routing (W3); verify calibration across shards.

**Data / agnostic:**
- Non-32×32 inputs → adaptive pooling (W12) required or FC sizes break.
- Imbalanced/overlapping classes → class-isolated sharding assumes clean separation; `_apply_asymmetric_splitting` is a stub. Document the assumption or implement splitting (out of scope for the exact-unlearning result; note as a limitation).

**Proof / determinism:**
- GPU non-determinism may prevent bit-identical weights → if parameter distance isn't ~0, fall back to a **distributional** argument (multiple seeds) and rely on prediction-agreement + MIA. State this.
- The scratch reference MUST replicate the exact slice schedule, replay, head sizing, and seed — otherwise the comparison is invalid.

---

## 8. Config parameter guidance (`config.py`)
- Keep: `NUM_SHARDS`, `NUM_SLICES_PER_SHARD` (architecture), normalization-from-metadata.
- Set by Optuna (W13), then freeze: `LEARNING_RATE`, `WEIGHT_DECAY`, `LABEL_SMOOTHING`, `REPLAY_RATIO`, `BATCH_SIZE`, `FC_LAYER_DROPOUT`, epochs/patience.
- Reconcile: `train_model.py` comment says "0.0 label smoothing during unlearning" but `UNLEARNING_LABEL_SMOOTHING=0.05` — use one value, identical in train/retrain/scratch.
- Remove/replace: hard-coded `FC_LAYER_1_INPUT=2048` (use adaptive pooling), any "kept for reference only" params, threshold knobs used as unlearning evidence.
- Add: `DATASET`, `NUM_CLASSES` (or derive from metadata), `INPUT_SIZE`, `IN_CHANNELS`, `SEED`, `MODEL_NAME` (for the factory), `ROUTING_MODE` (self-route vs gate).

---

## 9. Suggested implementation order (the "tomorrow" plan)
1. **W1 seeding** (unblocks everything; required for the proof).
2. **W2 canonical split** (all later numbers depend on it).
3. **W12 agnostic + adaptive pooling** (do early so the model is stable; enables extra datasets later).
4. **W3 self-routing head** + **W4 simple replay** (Lane A) — the accuracy/exactness core.
5. **W5 drop threshold / router unlearning** + **W6 head sizing**.
6. **W7 metrics as JSON** + **W10 logging** (kill the scraping coupling).
7. **W8 exactness proof harness** (scratch reference + tests + plots) — the headline deliverable.
8. **W9 viz fixes** + **W11 dead-code removal** + **W14 edge-case guards**.
9. **W13 Optuna once**, freeze, retire.
10. Add novelty comparisons/baselines (Section 5) as data/experiments.

---

## 10. Definition of Done (acceptance checklist)
- [ ] Same seed ⇒ identical runs (W1).
- [ ] Standard CIFAR-10 test set used; no leakage (W2).
- [ ] System works with no learned router, or the router is retrained on deletion; no threshold in the exactness claim (W3/W5).
- [ ] Simple seeded replay (Lane A) or adapter-based (Lane B) (W4).
- [ ] Deleted class has no active output; matches scratch head sizing (W6).
- [ ] All reported metrics computed and stored as JSON; no log scraping (W7).
- [ ] Scratch reference + five exactness metrics + MIA AUC≈0.5 + four plots (W8).
- [ ] Viz bugs fixed; single inference pass; no duplicate defs (W9).
- [ ] stdlib logging; runs from any directory (W10).
- [ ] No dead code / decorative params (W11).
- [ ] Runs on a second dataset by config only; adaptive pooling in place (W12).
- [ ] Optuna run once on validation, frozen into config, retired to `tuning/` (W13).
- [ ] Every Section 7 edge case handled or asserted (W14).
- [ ] ARCANE + RecEraser cited and compared; ≥1 modern class-unlearning baseline (Section 5).

---

## 11. References
- Bourtoule et al., *Machine Unlearning* (SISA), IEEE S&P 2021 — https://arxiv.org/pdf/1912.03817
- ARCANE: An Efficient Architecture for Exact Machine Unlearning, IJCAI 2022 — https://www.ijcai.org/proceedings/2022/0556.pdf
- RecEraser: Recommendation Unlearning, WWW 2022 — https://arxiv.org/pdf/2201.06820
- Towards Scalable Exact Machine Unlearning Using PEFT (S3T) — https://arxiv.org/html/2406.16257v2
- Machine Unlearning Fails to Remove…, ICLR 2025 (why accuracy≈random is insufficient) — https://proceedings.iclr.cc/paper_files/paper/2025/file/7e810b2c75d69be186cadd2fe3febeab-Paper-Conference.pdf
- Towards Reliable Empirical Machine Unlearning Evaluation (MIA) — https://arxiv.org/abs/2404.11577
- Boundary Unlearning, CVPR 2023 (class-unlearning baseline) — https://www.chenwang.net.cn/publications/Boundary-Unlearning-CVPR23.pdf
- Selective Synaptic Dampening (SSD) baseline — https://www.researchgate.net/publication/379278631
- Your paper — https://arxiv.org/abs/2604.27804

---

## 12. Self-audit of this plan (gaps, assumptions, risks)

**Confirmed complete:** covers correctness bugs (W2,W9,W11,W14), reproducibility (W1,W7,W10), the exactness path (W3–W6,W8), agnostic refactor (W12), tuning (W13), novelty (5), and the proof (4). Each workstream has acceptance criteria and a suggested order.

**Open decisions that must be made before coding (do not let the coding session guess):**
1. **Lane A vs Lane B** (3.3) — from-scratch CNN + simple replay, or frozen backbone + adapters. Affects W4 and accuracy ceiling. *Default: A.*
2. **Head sizing** (3.4) — fixed head + masking vs dynamic reduced head. *Default: dynamic (option b).*
3. **Routing** (3.2) — self-routing (default) vs retrained gate fallback. Confirm the accuracy of self-routing before removing the gate entirely; keep the gate code until W3 acceptance passes.

**Assumptions to verify on day one:**
- Class-sequential slicing truly makes each class a contiguous slice block in the current data (spot-check a shard's `slice_*_y.npy`). The exactness argument depends on it.
- The pre-unlearning checkpoint at `first_affected_slice-1` never saw the class via replay (true under class-sequential, but assert it in a test).
- CIFAR-10 is balanced enough that class-isolated sharding needs no splitting; for other datasets W12 + the splitting stub must be revisited.

**Known residual risks (not fixable by code alone):**
- **Novelty** (Section 5) — even fully implemented, competitiveness at a selective venue needs a new idea + baselines + possibly larger-scale data. Decide the target venue accordingly.
- **Exact bit-identity** may be unattainable on GPU; the distributional + MIA fallback (4.4/Section 7) is the mitigation — make sure the paper claims exactness at the right granularity.
- **"Unlearning by construction"** critique — class-isolated sharding makes deletion easy; consider addressing the harder distributed-class case as future work or a second experiment.

**Possibly missing / to add if scope allows:**
- A tiny **unit-test suite** asserting: post-deletion no slice contains the class; retrain starts at the right checkpoint; deleted class has zero support; replay is class-clean. (Cheap, high-value; recommend adding as `tests/test_unlearning_logic.py`.)
- A **sequential-deletion** experiment (delete several classes one after another) to show exactness and efficiency hold over a stream.
- A **requirements.txt / environment pin** for reproducibility (torch, torchvision, optuna, sklearn, seaborn versions).
