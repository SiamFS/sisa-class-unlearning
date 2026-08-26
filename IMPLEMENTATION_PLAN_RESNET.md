# Residual Backbone — Implementation Plan (W28–W29)

**Branch:** `resnet-backbone` (from `fix` @ `b8db808`)
**Extends** `IMPLEMENTATION_PLAN.md` (W1–W17) and `IMPLEMENTATION_PLAN_ACCURACY.md` (W18–W27).

**Framing:** this is an **architecture ablation**, not an upgrade. The question it answers is
whether the unlearning guarantees and efficiency characteristics are properties of the framework
or of one backbone. See `docs_architecture_ablation.pdf` for the paper-facing rationale.

---

## 1. Why (measured, not assumed)

**Parameters per training sample within a shard** — the quantity that matters, since class-isolated
sharding gives each specialist only `|C_k| * n / N_C` samples:

| model | params | CIFAR-10 (20,250/shard) | CIFAR-100 (2,835) | Tiny-IN (3,600) |
|---|---|---|---|---|
| SISAConvNet (current) | 621,313 | 31 | 219 | 173 |
| **ResNet-20** | **272,474** | **13** | **96** | — |
| ResNet-26 (4-stage, 64px) | 1,093,848 | — | — | 304 |
| ResNet-18 (ImageNet) | 11,173,962 | 497 | 3,547 | 3,104 |

ResNet-20 has **fewer parameters than the current CNN** while adding residual connections: the
current 0.62M sits mostly in `Linear(2048, 256)`, which global average pooling removes entirely.

**Gate sizing.** The current gate is **544,258 params — 200% of a ResNet-20 specialist**. The router
is twice the size of the model it routes to, and it is 2 conv layers sized for a 2-way decision;
at K=13 or K=25 that will not hold up.

**Not viable:** torchvision's `mobilenet_v3_small` / `shufflenet_v2_x0_5` / `squeezenet1_1` /
`mnasnet0_5` are ImageNet designs (stride-2 stems, 1.2–2.5M params). They run at 32×32 but discard
the image before learning, and are all *larger* than a ResNet-20 specialist. Their value comes from
pretrained weights, which reintroduce the Lane B exactness caveat (main plan §3.3).

**Not viable:** a linear router. MoE routers are linear because a *shared backbone* extracts features
first; here every shard is independent (exactness requires it), so the gate must extract its own
features from raw pixels.

---

## 2. Code contracts that MUST NOT break

The codebase encodes the classifier's position structurally. Every one of these must keep working:

| contract | used by |
|---|---|
| `fc_layer.5.weight` state-dict key | `create_model.py:233` (num_classes inference), `sisa_unlearning.py:808` (`_resize_model_head`) |
| `fc_layer.5.bias` present ⇒ linear head; `fc_layer.5.scale` ⇒ cosine | `create_model.py:240-242` |
| `fc_layer[:5](x)` returns penultimate features | `create_model.py:107`, `gate_free_routing_probe.py:63` |
| `fc_layer[5]` is the classifier module | `create_model.py:119`, `gate_free_routing_probe.py:64,306,355,392` |
| `model.classifier_type` attribute | `sisa_unlearning.py:817` |
| `model.penultimate(x)`, `model.routing_cosine(x)` | `plots.py` (`_shard_routing_cosines`, `_route_via_cosine`) |
| `create_sisa_model(num_classes, in_channels, classifier_type)` | `train_model.py:149`, `sisa_unlearning.py:816` |

### 2.1 The design that preserves all of them

`SISAResNet.fc_layer` mirrors `SISAConvNet.fc_layer`'s six-element shape, padding the positions a
residual net does not need:

```
[0] Identity          (SISAConvNet: Dropout)
[1] Identity          (SISAConvNet: Linear 2048->256)   <- absence of fc_layer.1.weight
[2] Identity          (SISAConvNet: BatchNorm1d)            identifies the architecture
[3] Identity          (SISAConvNet: ReLU)
[4] Dropout(RESNET_DROPOUT)
[5] Linear | CosineClassifier                            <- the classifier, position preserved
```

Consequences, all free:

* `fc_layer[:5]` is an identity chain, so `penultimate()` returns the trunk's GAP output unchanged.
* `fc_layer.5.weight` exists with shape `(num_classes, width)`, so W6 head resizing and the loader's
  class-count inference work untouched.
* `routing_cosine()` and the W21 cosine head work untouched.
* **Architecture is detectable from the checkpoint**: `SISAConvNet` has `fc_layer.1.weight`
  (the 2048→256 Linear); `SISAResNet` does not, because `Identity` has no parameters. No new
  metadata field, and old checkpoints keep loading.

---

## 3. W28 — Resolution-derived residual backbone

### 3.1 Depth follows from input resolution

Stages are added until the feature map reaches 8×8 before global average pooling:

```
n_stages = ceil(log2(input_size / RESNET_TARGET_SPATIAL)) + 1
widths   = [RESNET_BASE_WIDTH * 2**i for i in range(n_stages)]
depth    = 2 * RESNET_BLOCKS_PER_STAGE * n_stages + 2
```

| resolution | stages | widths | depth @3 blocks |
|---|---|---|---|
| 32×32 | 3 | [16, 32, 64] | **20** (= ResNet-20 exactly) |
| 64×64 | 4 | [16, 32, 64, 128] | 26 |

At 32×32 this reproduces He et al.'s ResNet-20 exactly, so the ablation is not a departure from
the standard architecture — it is the standard architecture with its resolution assumption made
explicit, consistent with how K, `S_k` and `S_max` are already derived.

### 3.2 Files and changes

**`training/create_model.py`**
* `BasicBlock` — 3×3 / 3×3 with a 1×1 projection shortcut on stride or width change.
* `SISAResNet(num_classes, in_channels, input_size, blocks_per_stage, base_width, classifier_type)`
  — `conv_layer` = stem + stages + `AdaptiveAvgPool2d(1)`; `fc_layer` per §2.1; `penultimate()` and
  `routing_cosine()` copied in form from `SISAConvNet` (identical bodies, so behaviour is shared).
* `create_sisa_model(...)` — dispatch on `config.MODEL_ARCH`; signature unchanged for callers.
* `load_model_complete(...)` — detect architecture via `'fc_layer.1.weight' in state_dict`
  (see §2.1) and build the matching class. Keeps the existing head-type and class-count inference.

**`config.py`**
```python
MODEL_ARCH              = 'convnet'   # 'convnet' | 'resnet'  (ablation switch)
RESNET_BLOCKS_PER_STAGE = 3           # 3 -> ResNet-20 at 32x32
RESNET_BASE_WIDTH       = 16
RESNET_TARGET_SPATIAL   = 8           # feature-map size before global average pooling
RESNET_DROPOUT          = 0.0         # CIFAR ResNets use none; see §3.3
def get_input_size(metadata_path=None) -> int:   # mirrors get_num_classes()
```

**`data_processing/entry_data_processing.py`** — record `input_size` in `sisa_data/metadata.json`
(the trailing spatial dim of `x_train`). The resolution must come from the data, not a constant.

### 3.3 Hyperparameters that MUST move with the architecture

* **`FC_LAYER_DROPOUT = 0.5` must not apply.** It is currently applied to a 2048-dim flattened
  feature, which is reasonable. A residual net's pre-classifier feature is **64-dim** after GAP;
  dropping half of 64 is destructive, and CIFAR ResNets conventionally use no dropout.
  `RESNET_DROPOUT = 0.0` is a separate constant so the ConvNet path is untouched.
* **`FC_LAYER_1_INPUT` / `FC_LAYER_1_HIDDEN` are ConvNet-only** and must not be read by the ResNet
  path — width comes from `base_width * 2**(n_stages-1)`.
* **Optimizer stays AdamW for the first run.** CIFAR ResNets conventionally use SGD+momentum with
  cosine decay and often gain 1–2 points, but changing optimizer and architecture together makes
  the ablation uninterpretable. A/B it *after* (§6).

### 3.4 Acceptance

* At 32×32 with defaults, parameter count matches ResNet-20 (~272k) and depth is 20.
* `fc_layer.5.weight` present; `fc_layer.1.weight` absent.
* `_resize_model_head` transplants the trunk and reinitializes only `fc_layer.5.weight`.
* An existing `SISAConvNet` checkpoint still loads while `MODEL_ARCH='resnet'`.
* `penultimate()` returns `base_width * 2**(n_stages-1)` features; `routing_cosine()` is in [-1, 1].
* All three routing modes (`gating`, `cosine`, `ensemble`) run.

---

## 4. W29 — Gate on the same family

One architecture family, two depths: specialists at `RESNET_BLOCKS_PER_STAGE`, gate at
`GATING_BLOCKS_PER_STAGE`. Measured at 1 block/stage, base 16:

| dataset | K | depth | params | MACs | vs specialist | vs current gate |
|---|---|---|---|---|---|---|
| CIFAR-10 | 2 | 8 | 77,522 | 14.6M | 28% | **7.0× smaller** |
| CIFAR-100 | 13 | 8 | 78,237 | 14.6M | 29% | — |
| Tiny-ImageNet | 25 | 10 | 310,761 | 77.3M | 28% | — |

**`training/create_model.py`** — `create_gating_model(...)` builds a `SISAResNet` trunk with
`num_classes=num_shards` when `MODEL_ARCH='resnet'`. `GatingNetwork` stays for the ConvNet path.
The existing checkpoint-compat logic (pool size from `fc1.weight`, `input_size` from metadata)
applies only to `GatingNetwork`; a ResNet gate is detected the same way as a ResNet specialist.

**`config.py`** — `GATING_BLOCKS_PER_STAGE = 1`, `GATING_BASE_WIDTH = 16`.

**Note:** `GATING_INPUT_SIZE` / `GATING_POOL_SIZE` (W24) are `GatingNetwork`-only. A ResNet gate
derives its own stages from resolution, so those knobs do not apply to it.

**Acceptance:** gate is ~28% of a specialist's parameters; routing accuracy does not regress from
the 94.63% baseline at K=2; the gate retrains cleanly on unlearning.

---

## 5. Migration and compatibility

* `MODEL_ARCH='convnet'` is the **default** — nothing changes until the flag is flipped, so the
  83.26% baseline stays reproducible on this branch.
* Existing checkpoints keep loading in both settings (§2.1 detection).
* **Switching `MODEL_ARCH` requires a full retrain**; the two architectures' checkpoints are not
  interchangeable. Data does not need regenerating unless `input_size` is missing from metadata.
* `tuning/tune.py` suggests `FC_LAYER_DROPOUT`, which is ConvNet-only. It must skip that parameter
  under `MODEL_ARCH='resnet'` or it will tune a value that has no effect.

---

## 6. Order of work

1. **W28** backbone + config + metadata `input_size`, `MODEL_ARCH='convnet'` (no behaviour change).
2. Verify §3.4 acceptance with the flag off, then on, without training.
3. **Full CIFAR-10 run** at `MODEL_ARCH='resnet'`, everything else fixed. This isolates the
   architecture effect against the 83.26% / 87.53%-ceiling baseline.
4. **W29** gate, re-measure routing.
5. **Then** the tuning A/Bs, one at a time: optimizer (SGD+momentum+cosine vs AdamW), then label
   smoothing (0.15 vs 0.05).

---

## 7. TODO

- [ ] **W28a** `BasicBlock` + `SISAResNet` in `training/create_model.py`
- [ ] **W28b** config: `MODEL_ARCH`, `RESNET_*`, `get_input_size()`
- [ ] **W28c** `create_sisa_model` dispatch; `load_model_complete` architecture detection
- [ ] **W28d** record `input_size` in metadata during data processing
- [ ] **W28e** `tuning/tune.py` skips `FC_LAYER_DROPOUT` under `MODEL_ARCH='resnet'`
- [ ] **W29a** ResNet gate + `GATING_BLOCKS_PER_STAGE` / `GATING_BASE_WIDTH`
- [ ] **W29b** verify gate checkpoint round-trip and unlearning retrain

---

## 8. Deferred / open

* **Blocks per stage is not yet derived from shard size.** At Tiny-ImageNet the 4-stage variant is
  **304 params/sample — worse than the current CNN's 173**. Depth grows with resolution while shards
  shrink with class count, so 64×64 needs `RESNET_BLOCKS_PER_STAGE = 2` (195/sample) or lower. Left
  as a config constant with a `validate_partition`-style warning rather than auto-derived, since
  varying depth per shard would complicate the W8 comparison.
* **WRN-16-4** (2.78M) as a CIFAR-10-only ablation for ARCANE comparability. Far too large for
  CIFAR-100/Tiny-ImageNet shards (882 params/sample).
* **BatchNorm is retained deliberately.** BN normally degrades class-incremental learning through
  cross-task statistics bias, but complete balanced replay (`MAX_REPLAY_SAMPLES_PER_CLASS = 5000`
  ≥ per-class count for all three datasets) removes that failure mode. This holds **only while
  replay stays complete**; a dataset too large for full replay would need GroupNorm or Continual
  Normalization. State this in the paper's limitations.
* **Accuracy expectation is an estimate, not a prediction.** Projected +2 to +5 points system
  accuracy, most likely ~+3, based on the parameter-budget argument and published ResNet-20 numbers
  on full CIFAR-10. Nobody has measured CIFAR ResNets under class-sequential slicing with balanced
  replay, so the transfer is unverified.
