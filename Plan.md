# Plan

This plan turns the thesis proposal into a concrete research and implementation roadmap. The goal is to build a reproducible pipeline that compares CL, MIM, and hybrid CL+MIM Vision Transformers on spurious-correlation robustness, then explains the performance differences through layer-wise representation geometry.

## Scope

- In: Waterbirds as the primary benchmark, CelebA as the second benchmark for a non-background spurious attribute, CL/MIM/CL+MIM ViT comparison, fine-tuning, average accuracy, worst-group accuracy, optional background consistency for Waterbirds, attention analysis, spectral analysis, object/spurious-attribute probing, correlation analysis, and thesis writing.
- Out: inventing a new robust training algorithm, large-scale SSL pretraining from scratch unless pretrained checkpoints are unavailable, causal claims stronger than the empirical evidence supports, and broad benchmarking beyond the thesis question.

## Final Research Goal

Answer the following core question:

> Do self-supervised pretraining objectives shape ViT representation geometry in ways that predict or explain robustness to spurious correlations?

The expected final thesis contribution is not just a table of accuracies. It should connect three levels of evidence:

1. Downstream robustness: average accuracy and worst-group accuracy.
2. Internal geometry: attention distance, attention homogeneity, spectral diversity, and effective rank.
3. Semantic accessibility: layer-wise object-label and spurious-attribute probing.

## Step 1: Freeze The Research Protocol ✓

Before implementing more code, write a short protocol document in `docs/experimental_protocol.md`.

Actions:

- Define the exact experimental conditions:
  - CL ViT, for example MoCo v3-style ViT.
  - MIM ViT, for example MAE.
  - Hybrid CL+MIM ViT, for example CMAE.
- Decide the primary backbone size, preferably `vit_base_patch16_224` to match the current configs.
- Decide which layers will be analyzed. Start with `[0, 3, 6, 9, 11]`, already present in `configs/analyze.yaml`.
- Define the default fine-tuning protocol:
  - Same image resolution.
  - Same optimizer.
  - Same learning-rate schedule.
  - Same number of epochs.
  - Same random seeds.
  - Same early-stopping/checkpoint rule.
- Define the primary metrics:
  - Average accuracy.
  - Group accuracy for all label/background groups.
  - Worst-group accuracy.
  - Optional background consistency.
  - Probe accuracy for object and spurious-attribute prediction.
  - Attention distance and attention homogeneity.
  - Singular value spectrum and effective rank.
- Define the minimum acceptable experiment set:
  - CL, MIM, CL+MIM on Waterbirds.
  - At least 3 seeds if compute allows.
  - At least 1 fully analyzed checkpoint per model family.

Deliverable:

- A frozen experimental protocol that can be referenced later in the thesis methodology chapter.

## Step 2: Make The Repository Reproducible ✓

The current repository has the right module structure, but most files are still placeholders. First make the project runnable end to end with a minimal baseline.

Actions:

- Add a `docs/` directory for protocol notes, dataset setup, and experiment logs.
- Add `outputs/`, `data/`, and checkpoint directories to `.gitignore` if not already ignored.
- Create `src/utils/config.py` for loading and validating YAML configs.
- Create `src/utils/seed.py` for deterministic seed setup.
- Create `src/utils/logging.py` or a lightweight logging helper for experiment metadata.
- Add a small smoke-test path that can run on CPU with a tiny subset of data.
- Update `README.md` whenever a workflow becomes real rather than aspirational.

Suggested validation:

- `python main.py --config configs/train.yaml --mode train` should fail only because data/checkpoints are missing, not because imports or config parsing are broken.
- Add at least one minimal test command once the first real functions exist.

Deliverable:

- A repo that can execute the main entry points and produce understandable errors when required assets are missing.

## Step 3: Prepare Waterbirds Correctly

Waterbirds is the central dataset because it exposes both object labels and a background-based spurious attribute. The dataset implementation must be exact, because all robustness claims depend on group labels.

Actions:

- Complete `src/datasets/waterbirds.py`.
- Verify the expected columns in `metadata.csv`:
  - `img_filename`
  - `y`
  - `place`
  - `split`
- Return a structured sample, preferably including:
  - `image`
  - `target`
  - `spurious`
  - `group`
  - `index`
  - optional `path`
- Define group IDs from `(target, spurious)`:
  - landbird on land.
  - landbird on water.
  - waterbird on land.
  - waterbird on water.
- Implement `get_waterbirds_dataloaders(config)` with train/val/test loaders.
- Use train transforms for fine-tuning and deterministic eval transforms for validation/test.
- Add a script or notebook to print group counts per split.

Validation:

- Confirm that the minority groups match the Waterbirds benchmark expectation.
- Confirm that every split has nonzero samples.
- Save a table of group counts for the thesis dataset section.

Deliverable:

- Reliable Waterbirds dataloaders and verified group metadata.

## Step 4: Add CelebA As A Secondary Dataset Only After Waterbirds Works

CelebA is useful for generalization, but it should not delay the core thesis. Treat it as Phase 2.

Actions:

- Create `src/datasets/celeba.py`.
- Use a target/spurious pair from the literature, such as Blond Hair as target and Male as spurious attribute.
- Match the same dataset interface used by Waterbirds:
  - `image`
  - `target`
  - `spurious`
  - `group`
  - `index`
- Add config support for switching datasets.

Validation:

- Print group counts.
- Verify that worst-group accuracy is computed identically to Waterbirds.

Deliverable:

- Secondary robustness benchmark, included only if the Waterbirds pipeline is stable.

## Step 5: Select And Normalize Pretrained Checkpoints

The comparison is only fair if the model wrappers expose the same backbone interface and the downstream heads are trained under the same protocol.

Actions:

- Identify available pretrained checkpoints:
  - MoCo v3 ViT for CL.
  - MAE ViT for MIM.
  - CMAE or another hybrid CL+MIM checkpoint for hybrid.
- Record checkpoint source, pretraining dataset, model size, patch size, and license.
- Prefer checkpoints with the same or very similar backbone:
  - ViT-Base.
  - patch size 16.
  - image size 224 or compatible.
- If exact matching is impossible, document the mismatch and control what can be controlled.
- Implement state-dict conversion utilities for each checkpoint family.
- Save converted checkpoints in a consistent local structure, for example:
  - `checkpoints/cl/model.pth`
  - `checkpoints/mim/model.pth`
  - `checkpoints/hybrid/model.pth`

Decision rule:

- If a suitable hybrid checkpoint is available, use it.
- If not, train or reproduce a smaller hybrid model only if compute allows.
- If hybrid reproduction is infeasible, narrow the thesis to CL vs. MIM and frame hybrid as planned future work or a limited pilot.

Deliverable:

- Three loadable model conditions, or a documented fallback if the hybrid checkpoint is not feasible.

## Step 6: Implement Model Wrappers And Layer Extraction

The current `src/models/cl_vit.py` is a start, but the thesis needs a unified model API across CL, MIM, and hybrid models.

Actions:

- Create or complete:
  - `src/models/cl_vit.py`
  - `src/models/mim_vit.py`
  - `src/models/hybrid_vit.py`
  - `src/models/factory.py`
- Implement a common interface:
  - `forward(x)` for classification.
  - `forward_features(x)` for final representation.
  - `get_intermediate_layers(x, layers)` for hidden states.
  - optional `get_attention_maps(x, layers)` for attention metrics.
- Prefer timm hooks or model-native APIs over fragile manual edits.
- Ensure all wrappers expose the same output shapes.
- Add support for freezing the backbone, partial fine-tuning, and full fine-tuning if needed.

Validation:

- Run a dummy batch through all three models.
- Check that logits have shape `[batch_size, 2]` for Waterbirds.
- Check that layer features and attention maps are returned for the configured layers.

Deliverable:

- A model factory that can instantiate CL, MIM, and hybrid ViTs from config.

## Step 7: Implement Fine-Tuning

Fine-tuning is the bridge between pretrained representations and robustness metrics. It must be controlled and repeatable.

Actions:

- Complete `src/training/train.py`.
- Implement validation inside the training loop.
- Save:
  - best checkpoint by validation worst-group accuracy.
  - final checkpoint.
  - training curves.
  - full config snapshot.
- Track average accuracy and group accuracies during validation.
- Add command-line support for seed overrides.
- Add config fields for:
  - checkpoint path.
  - freeze mode.
  - learning rate.
  - weight decay.
  - epochs.
  - scheduler.
  - batch size.
  - output directory.
- Keep the first main protocol as standard ERM fine-tuning.
- Add Group DRO only as an optional control, not the main thesis condition, unless the thesis question changes.

Validation:

- Overfit a tiny subset to verify training works.
- Run one short Waterbirds training job for one model before launching the full grid.

Deliverable:

- Reproducible fine-tuning pipeline for all model families.

## Step 8: Implement Robustness Evaluation

This step produces the main results for RQ1.

Actions:

- Create `src/evaluation/performance.py`.
- Implement:
  - average accuracy.
  - per-group accuracy.
  - worst-group accuracy.
  - confusion matrix.
  - group-wise sample counts.
- Make evaluation independent from training so it can run on saved checkpoints.
- Save metrics as JSON and CSV.
- Print a compact table in the terminal.
- Add `main.py --mode evaluate` integration.

Validation:

- Compare computed group counts to the dataset verification script.
- Confirm worst-group accuracy is the minimum of the four group accuracies.
- Check that predictions and labels are aligned after batching.

Deliverable:

- Evaluation tables for each model and seed.

## Step 9: Add Background Consistency If Feasible

Background consistency is valuable but should be treated as optional because it may require counterfactual or paired images.

Actions:

- Decide what form is feasible:
  - Existing Waterbirds metadata only.
  - Synthetic background swaps.
  - Group-matched comparison without image editing.
- If using synthetic swaps, define a careful and limited protocol:
  - same foreground object.
  - changed background.
  - no leakage of labels from generation artifacts.
- If synthetic swaps are too expensive, use background consistency as a qualitative or exploratory analysis only.

Validation:

- Manually inspect examples.
- Report the limitation clearly in the thesis.

Deliverable:

- Either a background-consistency metric or a justified decision to omit it from main claims.

## Step 10: Build A Feature Extraction Cache

The analysis metrics should run on frozen representations without repeatedly recomputing full model passes.

Actions:

- Create `src/analysis/extract_features.py`.
- For each checkpoint and dataset split, save:
  - final CLS features.
  - selected layer CLS features.
  - selected layer patch-token features if needed.
  - labels.
  - spurious-attribute labels.
  - group IDs.
  - optional attention maps.
- Store features in a structured output path:
  - `outputs/{experiment}/features/{split}/layer_{k}.pt`
- Include config and checkpoint hash in metadata.

Validation:

- Load saved features and verify shapes.
- Check that labels/spurious attributes/groups have the same length as features.
- Compare a small direct forward pass to cached values.

Deliverable:

- Reusable feature cache for attention, spectral, and probing analyses.

## Step 11: Implement Attention Distance

Attention distance tests whether models aggregate information locally or globally.

Actions:

- Complete `src/analysis/attention.py`.
- Exclude or separately handle the CLS token because it has no spatial coordinate.
- Compute patch-grid coordinates from image size and patch size.
- For each layer/head, compute the attention-weighted spatial distance.
- Aggregate results by:
  - model family.
  - layer.
  - head.
  - group.
  - correct vs. incorrect prediction if useful.
- Save results as CSV.
- Plot attention distance across layers.

Validation:

- Test the distance matrix on a small artificial patch grid.
- Confirm distances are nonnegative and bounded by the image diagonal.
- Confirm local attention produces lower scores than uniform/global attention in a synthetic test.

Deliverable:

- Layer-wise attention-distance plots and tables for CL, MIM, and CL+MIM.

## Step 12: Implement Attention Homogeneity

Attention homogeneity tests whether many tokens attend to the same locations, which may distinguish global semantic abstraction from token-specific diversity.

Actions:

- Implement homogeneity in `src/analysis/attention.py`.
- Start with cosine similarity between flattened query-token attention distributions.
- Add normalized mutual information only if discretization is well justified.
- Compute homogeneity per layer and head.
- Compare homogeneity against:
  - worst-group accuracy.
  - object probe accuracy.
  - spurious-attribute probe accuracy.
- Save CSV outputs and plots.

Validation:

- Verify that identical attention maps produce high homogeneity.
- Verify that random diverse attention maps produce lower homogeneity.
- Check computational cost before running on the full dataset.

Deliverable:

- Layer-wise attention-homogeneity curves and comparison against robustness.

## Step 13: Implement Spectral Diversity Analysis

Spectral analysis tests whether robustness is associated with richer or more compact feature spaces.

Actions:

- Extend `src/analysis/spectral.py`.
- Compute singular values and effective rank for each model/layer/split.
- Compute spectra on:
  - all samples.
  - each group separately.
  - optionally correctly vs. incorrectly classified samples.
- Normalize features consistently before SVD.
- Save:
  - singular values.
  - effective rank.
  - plots of spectrum decay.
  - effective-rank tables.
- Use PCA or UMAP for qualitative visualization only, not as primary evidence.

Validation:

- Use `torch.linalg.svdvals` instead of deprecated APIs where possible.
- Check stability by subsampling the same number of samples per group.
- Avoid over-interpreting UMAP clusters.

Deliverable:

- Spectral diversity results across layers and model families.

## Step 14: Implement Object And Background Probing

Probing is central because it directly measures what information is linearly accessible in each layer.

Actions:

- Complete `src/analysis/probing.py`.
- Train separate probes for:
  - object label `y`.
  - spurious attribute, for example Waterbirds `place`.
  - group ID if useful.
- Use frozen features only.
- Use train/validation/test splits consistently.
- Report probe accuracy by layer and model family.
- Add regularization or early stopping to avoid overfitting.
- Save probe metrics as CSV.
- Plot:
  - object probe accuracy by layer.
  - spurious-attribute probe accuracy by layer.
  - object-minus-spurious probe gap by layer.

Validation:

- Verify probe labels are correct.
- Compare probe results against simple majority baselines.
- Repeat with at least 3 seeds if probe variance is high.

Deliverable:

- Layer-wise object/spurious separability evidence for RQ2 and RQ3.

## Step 15: Run The Minimum Viable Experiment Grid

Start small, then scale. Do not launch the full grid until one model works end to end.

Phase A: smoke experiments.

- One model.
- One seed.
- Tiny subset.
- Short training.
- Full pipeline from training to evaluation to analysis.

Phase B: Waterbirds main experiments.

- CL on Waterbirds.
- MIM on Waterbirds.
- CL+MIM on Waterbirds.
- Same fine-tuning protocol.
- Same seeds.
- Save all configs and outputs.

Phase C: extended experiments.

- Add more seeds.
- Add CelebA.
- Add Group DRO or another robust training control only if the main comparison is stable.
- Add background consistency only if feasible.

Suggested experiment matrix:

| Dataset | Model | Training | Seeds | Required |
| --- | --- | --- | --- | --- |
| Waterbirds | CL | ERM fine-tuning | 3 | Yes |
| Waterbirds | MIM | ERM fine-tuning | 3 | Yes |
| Waterbirds | CL+MIM | ERM fine-tuning | 3 | Yes |
| CelebA | CL | ERM fine-tuning | 1-3 | Optional |
| CelebA | MIM | ERM fine-tuning | 1-3 | Optional |
| CelebA | CL+MIM | ERM fine-tuning | 1-3 | Optional |
| Waterbirds | CL/MIM/Hybrid | Group DRO | 1 | Optional control |

Deliverable:

- A complete Waterbirds result table before adding optional experiments.

## Step 16: Analyze Correlations Between Geometry And Robustness

This step connects RQ1 to RQ2 and RQ3.

Actions:

- Create `src/analysis/correlate.py`.
- Merge all result files into one analysis table:
  - model family.
  - seed.
  - layer.
  - average accuracy.
  - worst-group accuracy.
  - attention distance.
  - attention homogeneity.
  - effective rank.
  - object probe accuracy.
  - spurious-attribute probe accuracy.
  - object-spurious probe gap.
- Compute rank correlations where sample size allows.
- Focus on effect patterns, not overclaiming statistical significance from small N.
- Compare early, middle, and late layers.
- Check whether robust models:
  - have higher object probe accuracy.
  - have lower spurious-attribute probe accuracy.
  - have a larger object-spurious gap.
  - show more global attention.
  - preserve useful spectral diversity.

Validation:

- Use bootstrap confidence intervals if enough runs exist.
- Report uncertainty across seeds.
- Avoid causal language unless the experiment supports it.

Deliverable:

- Main synthesis table connecting robustness and representation metrics.

## Step 17: Define Figure And Table Targets Early

The thesis should be written around a small number of strong figures, not a large pile of disconnected plots.

Target tables:

- Dataset group counts for Waterbirds and optional CelebA.
- Main downstream performance table:
  - average accuracy.
  - group accuracies.
  - worst-group accuracy.
- Model/checkpoint summary table.
- Correlation/synthesis table.

Target figures:

- Experiment pipeline diagram.
- Worst-group accuracy comparison across CL, MIM, and CL+MIM.
- Attention distance by layer.
- Attention homogeneity by layer.
- Effective rank by layer.
- Object vs. spurious-attribute probe accuracy by layer.
- Optional PCA/UMAP visualization for qualitative illustration.

Deliverable:

- A `figures/` or `outputs/final_figures/` directory with thesis-ready plots.

## Step 18: Write The Thesis In Parallel With Experiments

Do not wait until all experiments are finished to start writing. The introduction, related work, methodology, and dataset sections can be drafted early.

Suggested chapter workflow:

1. Introduction:
   - Spurious correlations.
   - Why average accuracy is insufficient.
   - Why ViTs and SSL objectives are interesting.
   - Research questions and contributions.
2. Related Work:
   - Spurious correlations and group robustness.
   - ViTs under spurious correlations.
   - CL vs. MIM representation structure.
   - Hybrid CL+MIM models.
3. Methodology:
   - Model families.
   - Datasets.
   - Fine-tuning protocol.
   - Robustness metrics.
   - Representation metrics.
4. Experiments:
   - Implementation details.
   - Main Waterbirds results.
   - Optional CelebA results.
5. Analysis:
   - Attention results.
   - Spectral results.
   - Probe results.
   - Correlation synthesis.
6. Discussion:
   - What evidence supports H1, H2, and H3.
   - What remains uncertain.
   - Limitations.
7. Conclusion:
   - Main empirical findings.
   - Practical implications for diagnosing SSL ViTs.
   - Future work.

Deliverable:

- A continuously updated thesis draft with result placeholders replaced as experiments finish.

## Step 19: Interpret Hypotheses Carefully

Map each hypothesis to the evidence needed to support or reject it.

H1:

- Claim: CL-pretrained ViTs may be more robust than pure MIM because CL learns more global, shape-oriented representations.
- Required evidence:
  - CL has higher worst-group accuracy than MIM.
  - CL has more global attention or stronger object probes in relevant layers.
  - Background probe accessibility is lower or less dominant than in MIM.

H2:

- Claim: Hybrid CL+MIM may provide a better trade-off between semantic abstraction and token diversity.
- Required evidence:
  - Hybrid matches or exceeds CL in worst-group accuracy.
  - Hybrid preserves higher spectral diversity or lower excessive attention homogeneity than CL.
  - Hybrid has strong object probes without disproportionately strong spurious-attribute probes.

H3:

- Claim: Robust models encode object information more strongly or more separably than spurious-attribute information in later layers.
- Required evidence:
  - Object probe accuracy exceeds spurious-attribute probe accuracy in later layers.
  - The object-spurious probe gap correlates with worst-group accuracy.
  - Layer-wise behavior differs systematically between robust and shortcut-prone models.

Deliverable:

- A final hypothesis-evidence matrix for the discussion chapter.

## Step 20: Manage Risks And Fallbacks

Risk: pretrained checkpoints are incompatible.

- Fallback: use closest available backbones and document differences.
- Fallback: reduce scope to CL vs. MIM if hybrid is infeasible.

Risk: compute is insufficient for full fine-tuning.

- Fallback: reduce seeds first, not the analysis quality.
- Fallback: use linear probing plus limited fine-tuning.
- Fallback: freeze most of the backbone for pilot experiments.

Risk: CelebA takes too long.

- Fallback: keep CelebA as a secondary validation or omit it from main claims.

Risk: attention extraction is difficult for a checkpoint implementation.

- Fallback: use timm-compatible models or hooks.
- Fallback: prioritize probing and spectral analysis, which require only hidden states.

Risk: correlations are weak or inconsistent.

- Fallback: this is still a valid empirical result.
- Frame the contribution as showing which proposed geometric signatures do or do not explain robustness.

Risk: UMAP/PCA visualizations look persuasive but are unstable.

- Fallback: use them only as qualitative illustrations.
- Keep quantitative claims based on accuracies, spectra, probes, and attention metrics.

Deliverable:

- A limitations section that is honest and technically defensible.

## Step 21: Suggested Implementation Order In The Existing Repo

Use this order to avoid building analysis code before the basic experiment pipeline works.

1. Complete config loading and seed setup.
2. Complete `src/datasets/waterbirds.py`.
3. Add `src/datasets/factory.py`.
4. Add model factory and finish CL model loading.
5. Add MIM and hybrid model wrappers.
6. Complete training and validation.
7. Implement `src/evaluation/performance.py`.
8. Run one CL Waterbirds smoke experiment.
9. Add feature extraction cache.
10. Complete probing analysis.
11. Complete spectral analysis.
12. Complete attention distance.
13. Complete attention homogeneity.
14. Run full Waterbirds experiment grid.
15. Add correlation/synthesis analysis.
16. Add optional CelebA.
17. Generate final figures.
18. Write results and discussion.

## Step 22: Weekly Milestone Plan

Week 1:

- Freeze protocol.
- Verify dataset setup.
- Make Waterbirds dataloader reliable.
- Make main entry points runnable.

Week 2:

- Implement model factory.
- Load at least one pretrained CL checkpoint.
- Run first fine-tuning smoke test.

Week 3:

- Add MIM and hybrid checkpoint loading.
- Complete evaluation metrics.
- Produce first Waterbirds performance table.

Week 4:

- Build feature extraction cache.
- Implement probing.
- Produce object/spurious probe plots.

Week 5:

- Implement spectral analysis.
- Implement attention distance.
- Produce first layer-wise representation plots.

Week 6:

- Implement attention homogeneity.
- Run full Waterbirds grid across seeds.
- Start correlation synthesis.

Week 7:

- Add CelebA or optional controls if Waterbirds results are stable.
- Otherwise strengthen Waterbirds analysis and rerun missing seeds.

Week 8:

- Finalize figures and tables.
- Write results, analysis, and discussion chapters.
- Check every claim against the available evidence.

## Step 23: Definition Of Done

The thesis project is ready to write up when the following are complete:

- Waterbirds dataloading is verified with group counts.
- CL, MIM, and CL+MIM models are loadable or fallback conditions are documented.
- Fine-tuning runs are reproducible from config files.
- Average accuracy and worst-group accuracy are computed for every required condition.
- Layer-wise features are cached for every required checkpoint.
- Object/spurious-attribute probes are trained and evaluated.
- Spectral diversity metrics are computed across layers.
- Attention distance and homogeneity are computed or explicitly scoped out with justification.
- Final plots and tables exist for the main claims.
- Hypotheses H1, H2, and H3 are each discussed using direct evidence.
- Limitations are explicit, especially around checkpoint comparability, compute, and causal interpretation.

## Open Questions

- Which exact pretrained checkpoints are available for CL, MIM, and CL+MIM with compatible ViT backbones?
- How much compute is available for full fine-tuning and multiple seeds?
- Should CelebA be treated as a required secondary benchmark or only as an optional extension after Waterbirds is complete?
