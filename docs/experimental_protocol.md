# Experimental Protocol

Status: ✓ Step 1 complete

This document freezes the initial experimental protocol for the thesis "Linking Self-Supervised Representation Geometry to Spurious-Correlation Robustness in Vision Transformers." It defines the model conditions, datasets, default training settings, metrics, and minimum experiment set before the implementation work continues.

## Research Aim

The thesis studies whether self-supervised pretraining objectives organize ViT representations in ways that are associated with robustness to spurious correlations. The core comparison is between a contrastive model, a masked image modeling model, and a hybrid model that combines both objectives.

The central question is:

> Do CL, MIM, and CL+MIM ViTs differ in worst-group robustness, and can those differences be connected to layer-wise representation geometry?

## Research Questions

RQ1: Do ViTs pretrained with CL, MIM, and hybrid CL+MIM differ in robustness to spurious correlations after fine-tuning?

RQ2: Which representation properties correlate with robustness: global vs. local attention, attention homogeneity, spectral diversity, or object/background separability?

RQ3: At which layers are object-related and spurious-attribute features encoded, amplified, or suppressed, and does this behavior differ between CL, MIM, and CL+MIM?

## Model Conditions

The default backbone is ViT-B/16 with 224 x 224 inputs.

| Condition | Method | Role in thesis | Preferred checkpoint |
| --- | --- | --- | --- |
| CL | MoCo v3 ViT-B/16 | Contrastive image-level SSL baseline | ImageNet-1K pretrained MoCo v3 ViT-B |
| MIM | MAE ViT-B/16 | Masked token-level SSL baseline | ImageNet-1K pretrained MAE ViT-B |
| CL+MIM | CMAE ViT-B/16 | Hybrid SSL condition | ImageNet-1K pretrained CMAE-Base |

The initial protocol uses pretrained checkpoints rather than reproducing large-scale SSL pretraining. If a compatible checkpoint is unavailable, the fallback order is:

1. Use the closest public ViT-B/16 checkpoint and document any mismatch.
2. Use a smaller compatible variant only for pilot experiments.
3. Reproduce pretraining at controlled scale only if compute allows.
4. If CMAE is infeasible, report CL vs. MIM as the main result and treat CL+MIM as a documented limitation or pilot.

## Primary Dataset

The primary benchmark is Waterbirds.

Required dataset sample fields:

- `image`: input image tensor.
- `target`: object class, landbird or waterbird.
- `spurious`: spurious attribute. For Waterbirds this is the `place` metadata column, land or water.
- `group`: the ordered pair `(target, spurious)`.
- `index`: dataset index.
- `path`: image path.

Waterbirds may also expose `place` and `background` aliases for interpretability, but shared training, evaluation, and analysis code should consume `spurious`.

Required Waterbirds groups:

| Group | Object | Background | Expected role |
| --- | --- | --- | --- |
| 0 | landbird | land | majority/aligned |
| 1 | landbird | water | minority/conflicting |
| 2 | waterbird | land | minority/conflicting |
| 3 | waterbird | water | majority/aligned |

Waterbirds is mandatory because it exposes explicit group labels and a spatially interpretable spurious attribute.

## Secondary Dataset

CelebA is the second benchmark. It should be added after the Waterbirds pipeline is stable, but its role is important: it tests whether the findings generalize beyond background-based spurious correlations.

Default target/spurious pair:

- Target: Blond Hair.
- Spurious attribute: Male.

CelebA should use the same evaluation interface as Waterbirds: image, target, spurious label, group ID, split, and index.

## Compute Environment

The intended training environment is a V100 GPU cluster. Effective batch-size defaults may require gradient accumulation on V100 hardware.

## Fine-Tuning Defaults

The default fine-tuning recipe follows the ViT fine-tuning setup from the CL-vs-MIM analysis protocol shown in the proposal reference material, based on Xie et al. for fine-tuning-style settings.

| Parameter | Default |
| --- | --- |
| Optimizer | AdamW |
| Base learning rate | 1.25e-3 |
| Weight decay | 0.05 |
| Effective batch size | 2048 |
| Epochs | 100 |
| Learning-rate schedule | cosine |
| Warmup epochs | 20 |
| Warmup schedule | linear |
| RandAugment | 9, 0.5 |
| Label smoothing | 0.1 |
| Mixup alpha | 0.8 |
| CutMix alpha | 1.0 |
| Stochastic depth | 0.1 |
| Layer-wise learning-rate decay | 0.65 |
| Gradient clipping | 5.0 |

If hardware does not support the effective batch size, use gradient accumulation and record the actual per-device batch size. If gradient accumulation is still infeasible, reduce the effective batch size and treat the run as a resource-adjusted setting in the experiment log.

The default downstream training objective is standard ERM fine-tuning. Group DRO or other robust objectives are optional controls, not the main thesis condition.

## Linear Probing Defaults

Linear probes are used for representation analysis, not as the primary downstream robustness metric.

| Parameter | Default |
| --- | --- |
| Optimizer | SGD |
| Base learning rate | 1.0 |
| Weight decay | 0.05 |
| Effective batch size | 1024 |
| Epochs | 50 |
| Learning-rate schedule | cosine |
| Warmup epochs | 0 |
| Augmentation/regularization | disabled unless explicitly stated |

The probe protocol must be identical for object-label probes and background-label probes.

## Pretraining/Reproduction Defaults

Large-scale SSL pretraining is out of scope by default. These settings apply only if a checkpoint must be reproduced at controlled scale, especially for a missing hybrid checkpoint.

| Parameter | Default |
| --- | --- |
| Optimizer | AdamW |
| Base learning rate | 1.0e-4 |
| Weight decay | 0.05 |
| Effective batch size | 1024 |
| Epochs | 100 |
| Learning-rate schedule | multistep |
| Warmup epochs | 10 |
| Warmup schedule | linear |
| RandAugment | 9, 0.5 |
| Label smoothing | 0.1 |
| Mixup alpha | 0.8 |
| CutMix alpha | 1.0 |
| Stochastic depth | 0.1 |
| Layer decay | 1.0 |
| Gradient clipping | 5.0 |

For MAE-specific reproduction, preserve MAE's defining design choices: random patch masking, asymmetric encoder-decoder, and discarding the decoder for downstream recognition. The default mask ratio should be 75% unless the selected checkpoint or reproduction code specifies otherwise.

For CMAE-specific reproduction, preserve the defining hybrid structure: an online masked autoencoding branch and a momentum contrastive branch. The protocol should use an official implementation or checkpoint if possible.

## Splits And Seeds

Minimum required Waterbirds setting:

- Train/validation/test splits from the official metadata.
- Three random seeds if compute allows: `42`, `43`, `44`.
- If compute is limited, run one complete seed first and add more seeds after the full pipeline is validated.

For all experiments, save:

- config file.
- random seed.
- checkpoint source.
- git commit hash if available.
- output directory.
- final metrics.
- best-checkpoint selection rule.

## Layer Analysis Defaults

Default layers:

```text
[0, 3, 6, 9, 11]
```

These layers cover early, middle, and late transformer behavior in a 12-block ViT-B/16. If model wrappers expose layers with a different indexing convention, record the mapping explicitly.

Required extracted values:

- final CLS representation.
- selected layer CLS representations.
- selected layer patch-token representations when needed.
- attention maps for selected layers when available.
- target labels.
- spurious-attribute labels.
- group IDs.
- prediction correctness.

## Evaluation Metrics

Primary downstream metrics:

- Average accuracy.
- Per-group accuracy.
- Worst-group accuracy.

Optional downstream metric:

- Background consistency, only if a defensible paired or counterfactual protocol is available.

Representation metrics:

- Attention distance.
- Attention homogeneity, initially cosine similarity over attention distributions; normalized mutual information can be added if discretization is justified.
- Singular value spectrum.
- Effective rank.
- PCA or UMAP visualizations for qualitative inspection only.

Probe metrics:

- Object-label probe accuracy by layer.
- Spurious-attribute probe accuracy by layer.
- Object-minus-spurious probe gap by layer.

## Minimum Experiment Set

The minimum thesis-complete experiment set is:

| Dataset | Model | Training | Seeds | Required |
| --- | --- | --- | --- | --- |
| Waterbirds | MoCo v3 ViT-B/16 | ERM fine-tuning | 1-3 | Yes |
| Waterbirds | MAE ViT-B/16 | ERM fine-tuning | 1-3 | Yes |
| Waterbirds | CMAE ViT-B/16 | ERM fine-tuning | 1-3 | Yes, unless checkpoint infeasible |

For each required checkpoint, run:

1. Fine-tuning.
2. Test-set robustness evaluation.
3. Feature extraction.
4. Object/spurious probing.
5. Spectral analysis.
6. Attention analysis where attention maps are accessible.

Second-dataset and optional extensions:

| Dataset | Model | Training | Seeds | Priority |
| --- | --- | --- | --- | --- |
| CelebA | MoCo v3 / MAE / CMAE | ERM fine-tuning | 1-3 | Second benchmark |
| Waterbirds | MoCo v3 / MAE / CMAE | Group DRO | 1 | Low/Control |
| Waterbirds | MoCo v3 / MAE / CMAE | background consistency | 1 | Exploratory |

## Checkpoint Comparability Rules

The comparison is valid only if checkpoint differences are tracked.

Record for every checkpoint:

- method name.
- source URL or local source.
- pretraining dataset.
- architecture.
- patch size.
- input resolution.
- pretraining epochs, if known.
- checkpoint license.
- whether the checkpoint includes only encoder weights or a full pretraining model.

Preferred comparability:

1. Same backbone: ViT-B/16.
2. Same pretraining dataset: ImageNet-1K.
3. Same downstream fine-tuning recipe.
4. Same evaluation and analysis pipeline.

If pretraining epochs differ between methods, do not hide that difference. Report it as a limitation and avoid attributing all performance differences solely to objective type.

## Hypothesis-To-Evidence Map

H1: CL may be more robust than MIM for background-based spurious correlations because CL tends to learn global, shape-oriented representations.

Evidence needed:

- MoCo v3 has higher worst-group accuracy than MAE.
- MoCo v3 shows more global attention in later layers.
- MoCo v3 has stronger object probes or weaker spurious-attribute dominance than MAE.

H2: Hybrid CL+MIM may provide a better trade-off between global semantic abstraction and local token diversity.

Evidence needed:

- CMAE matches or exceeds MoCo v3 and MAE in worst-group accuracy.
- CMAE preserves spectral diversity or token-level diversity better than pure CL.
- CMAE has strong object probes without excessive spurious-attribute probe dominance.

H3: Robust models should encode object-related information more strongly or more separably than spurious-attribute information in later layers.

Evidence needed:

- Object probe accuracy exceeds spurious-attribute probe accuracy in later layers.
- The object-spurious probe gap aligns with worst-group accuracy.
- Layer-wise differences are systematic across model families and seeds.

## Implementation Targets

Config files should eventually expose the frozen protocol:

- `configs/train.yaml`: fine-tuning defaults.
- `configs/analyze.yaml`: layer and analysis defaults.
- `configs/models/*.yaml`: checkpoint-specific metadata, if needed.
- `configs/datasets/*.yaml`: dataset-specific paths and group definitions, if needed.

The first complete run should be:

```powershell
python main.py --config configs/train.yaml --mode train
python main.py --config configs/train.yaml --mode evaluate
python main.py --config configs/analyze.yaml --mode analyze
```

## Sources Used For This Protocol

- Park et al., "What Do Self-Supervised Vision Transformers Learn?", especially Appendix A training settings and Appendix C attention analysis.
- Chen, Xie, and He, "An Empirical Study of Training Self-Supervised Vision Transformers" / MoCo v3.
- He et al., "Masked Autoencoders Are Scalable Vision Learners" / MAE.
- Huang et al., "Contrastive Masked Autoencoders are Stronger Vision Learners" / CMAE.
- The proposal text in `proposal_text.txt`.

## Remaining Unknowns

- Exact local or remote checkpoint paths for MoCo v3, MAE, and CMAE.
- Available GPU memory and whether the effective batch-size defaults require gradient accumulation.
- Exact CelebA schedule after the Waterbirds pipeline is stable.
