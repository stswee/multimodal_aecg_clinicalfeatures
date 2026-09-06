# Multimodal Integration of Ambulatory ECG and Clinical Features for Sudden Cardiac Death and Pump Failure Death Prediction

This repository contains the analysis and plotting scripts for the MUSIC four-year sudden cardiac death (SCD) and pump-failure death (PFD) prediction workflows. Scripts assume local access to the MUSIC data and require path, environment, and hardware settings to be reviewed before execution.

## 0_Preprocessing

This folder preprocesses ambulatory ECG traces and computes HRV-ready intermediate files.

The notebook:

```bash
Preprocessing.ipynb
```

provides an example preprocessing workflow. To process the full ECG set, review paths and run:

```bash
run_preprocess_all_ecgs_HRV_complete.sh
```

## 1_Segmenting

This folder segments preprocessed ECGs into analysis windows.

First, review paths and run:

```bash
create_window_index_metadata.py
```

to generate `window_index_metadata.csv`. Then run one of:

```bash
run_segment_HRV_complete.sh
run_segment_HRV_complete_parallel.sh
```

Both scripts calculate HRV features for each ECG segment; the parallel launcher uses multiprocessing.

## 2_Labeling

This folder contains the MUSIC labeling notebook used to preprocess tabular data, generate the five outer folds, define four-year SCD/PFD labels, and create patient prompts.

Run:

```bash
Labeling_MUSIC_Updated.ipynb
```

after reviewing the path settings in the notebook.

## 3_ECG_Modeling

This folder trains and evaluates ECG-only models. See `3_ECG_Modeling/README.md` for the full ECG workflow, cohort checks, expected directory structure, and output descriptions.

After reviewing `wave1_capacity.sh`, run:

```bash
wave1_capacity.sh
```

After training completes, run:

```bash
calibrate_ecg_platt.py
select_ecg_thresholds.py
ecg_decision_curve_analysis.py
```

`ECG_Analysis.ipynb` reviews hyperparameter selection and pooled uncalibrated performance.

## 4_LLM_Modeling

This folder generates LLM risk assessments, embeds text with frozen biomedical encoders, trains text-only nested-CV classifiers, and produces post hoc evaluation outputs.

Run the workflow in this order after reviewing script paths and model settings:

```bash
run_generate_LLM_risks.sh
run_embed_llm_risks.sh
run_text_embeddings_nested_cv.sh
run_text_posthoc_evaluation.sh
```

`LLM_Response_Verification.ipynb` supports response inspection and verification.

## 5_Benchmarking

This folder runs structured clinical baseline models and matched MLP benchmarks.

Run:

```bash
run_tabular_nested_cv.sh
run_tabular_mlp_matched_nested_cv.sh
```

After model training, run:

```bash
tabular_posthoc_plots.py
```

Post hoc literature-reduced continuous-LVEF benchmark analyses are in:

```bash
supplementary/literature_reduced/
```

## 6_Multimodal_Modeling

This folder trains endpoint-specific multimodal fusion models using ECG representations, text representations, and structured clinical variables.

Run:

```bash
run_multimodal_nested_cv.sh
```

After training, run:

```bash
run_multimodal_posthoc_evaluation.sh
multimodal_text_shuffling_analysis.py
plot_multimodal_evaluation.py
```

Post hoc literature-reduced continuous-LVEF multimodal analyses are in:

```bash
supplementary/literature_reduced/
```

## 7_Comparative_Analysis

This folder contains the comparative analysis, manuscript figure/table generation, and multimodal ECG/text explanation scripts selected for release.

Run the endpoint-specific comparative analysis with:

```bash
run_endpoint_specific_comparative_analysis.sh
```

Generate manuscript figures and tables with:

```bash
run_generate_manuscript_figures_tables.sh
```

Generate patient-level multimodal ECG/text explanation artifacts with:

```bash
run_generate_multimodal_ecg_text_explanations.sh
```

## GitHub Release Package

This release package contains selected derived artifacts needed for reuse and review:
- ECG embeddings
- LLM-generated response CSVs
- patient-level prediction and classification CSVs
- final outer-fold model checkpoint artifacts
- release manifest, checksums, and provenance documentation
