MUSIC four-year endpoint-specific text modeling

Status

This directory contains the final text-generation, embedding, endpoint-specific
nested-cross-validation, and text-only posthoc workflows. The computational
text-only analysis is complete.

The final revision treats the outcomes as two independent binary tasks:

SCD: 577 survivors versus 71 sudden cardiac deaths (n = 648); 82 PFD
patients are excluded.

PFD: 577 survivors versus 82 pump-failure deaths (n = 659); 71 SCD
patients are excluded.

The competing endpoint is removed before classifier fitting, preprocessing,
calibration, threshold selection, and evaluation. The primary SCD text contains
only the SCD risk/rationale, and the primary PFD text contains only the PFD
risk/rationale. The old joint SCD+PFD text is retained only as a prespecified
ablation.

Do not mix final v3 outputs with text_nested_4year_v2.

Directory files

File

Purpose

generate_llm_risks.py

Generates and parses the three locked prompt conditions.

run_generate_LLM_risks.sh

Pins model revisions, verifies the prompt checksum, and launches both LLaMA models.

embed_llm_risks.py

Creates frozen BioBERT/ClinicalBERT embeddings for endpoint-specific, joint, neutral, and deterministic texts.

run_embed_llm_risks.sh

Launches endpoint-specific embedding extraction.

train_text_embeddings_nested_cv.py

Fits the independent SCD and PFD nested-CV text classifiers.

run_text_embeddings_nested_cv.sh

Runs both endpoints and combines their untouched predictions.

plot_text_evaluation.py

Produces pooled ROC, precision-recall, and calibration figures.

text_decision_curve_analysis.py

Produces exploratory text-only decision curves.

run_text_posthoc_evaluation.sh

Launches the CPU-only text posthoc analysis.

LLM_Results_Reporting.txt

Copy-ready methods, results, interpretation, and reviewer-response text.

Attribution analysis is not run here. Final attribution diagnostics are linked
to the selected endpoint-specific ECG + full LLM text multimodal checkpoints
and are run from 7_Comparative_Analysis.

Locked generation and embedding provenance

Item

Immutable value

Prompt CSV SHA-256

40644ed776353758c4c94bb752620f177244ac53e559d82538b7402a09f170e6

LLaMA-3.1-8B-Instruct revision

0e9e39f249a16976918f6564b8830bc894c89659

LLaMA-3.2-3B-Instruct revision

0cb88a4f764b7a12671c53f0838cd831a0843b95

BioBERT revision

924f12e0c3db7f156a765ad53fb6b11e7afedbc8

ClinicalBERT revision

d5892b39a4adaed74b92212a44081509db72f87b

Decoding

Greedy; do_sample=False; max_new_tokens=256; seed 42

Pooling

Frozen encoder [CLS] representation

Maximum encoder length

512 tokens

Locked nested-CV seed

42

The generation manifest records model/tokenizer revisions, package versions,
precision, device mapping, decoding settings, prompt checksum, and raw response.
Every embedding manifest records the source-response checksum, source columns,
patient/text hashes, resolved encoder revision, pooling, dimensions, and
truncation indicators.

Verified result-package checksums:

2ef36d5163a9f2180fb0049e3901aaa8271812a25e7dac1b63dea4bf557f6bd4  endpoint_specific_text_v3_results.tar.gz
31dcb7c26e52e6f6b03dd2ac86eb32ad1d084bc906b399a33afb491aaa0a18f3  endpoint_specific_text_v3_posthoc_results.tar.gz

Record the exact installed-script hashes after copying files to the server:

cd /home/sswee/multimodal_aecg_clinicalfeatures/4_LLM_Modeling

sha256sum \
  generate_llm_risks.py \
  run_generate_LLM_risks.sh \
  embed_llm_risks.py \
  run_embed_llm_risks.sh \
  train_text_embeddings_nested_cv.py \
  run_text_embeddings_nested_cv.sh \
  plot_text_evaluation.py \
  text_decision_curve_analysis.py \
  run_text_posthoc_evaluation.sh \
  README.md \
  LLM_Results_Reporting.txt \
  > installed_files.sha256

Final nine-arm analysis

Arm

Interpretation

full_risk_no_ecg

Primary endpoint-specific risk label plus rationale.

label_only_no_ecg

Endpoint-specific categorical risk label only.

rationale_only_no_ecg

Endpoint-specific rationale only.

joint_full_risk_no_ecg

Joint SCD+PFD response; cross-endpoint ablation only.

neutral_summary_no_ecg

Neutral LLM-generated summary without explicit risk reasoning.

deterministic_template_no_ecg

Fixed non-LLM verbalization of the baseline variables.

full_risk_with_ecg

Endpoint-specific response generated with ECG impressions.

label_only_with_ecg

Corresponding risk label only.

rationale_only_with_ecg

Corresponding rationale only.

This is the final text-representation ablation set. No further response
generation, embedding, or text-classifier ablation is planned.

Nested-CV design

Five locked outer folds and four outer-specific inner folds.

Separate SCD and PFD model selection.

Thirty-two primary candidates per endpoint/fold: two LLaMA sources, two
frozen encoders, and eight linear/MLP configurations.

Selection uses pooled inner out-of-fold ROC-AUC. Configurations within 0.005
of the maximum are ordered by fewer trainable parameters, higher PR-AUC, and
configuration name.

The fold-specific selected source, encoder, classifier, and optimization
configuration are held fixed across the nine representation arms.

Scaling, class weighting, early stopping, Platt calibration, and Youden
threshold selection use training data only.

Each eligible patient receives one untouched outer-test prediction per arm.

Five thousand patient-level bootstrap replicates are used for uncertainty.

Holm correction is applied across the prespecified text comparisons, both
endpoints, and all three metrics.

Output locations

/home/sswee/music/
├── llm_responses_4year_v2/
├── text_embeddings_4year_v3_endpoint_specific/
└── text_nested_4year_v3_endpoint_specific/
    ├── tasks/
    │   ├── scd/
    │   └── pfd/
    ├── combined_evaluation/
    ├── text_evaluation_figures_endpoint_specific/
    └── text_decision_curve_analysis_endpoint_specific/

Execution

The final results already exist; do not rerun completed stages unless an
artifact fails the checks below. For a clean reconstruction, run in this order:

cd /home/sswee/multimodal_aecg_clinicalfeatures/4_LLM_Modeling

chmod +x \
  generate_llm_risks.py run_generate_LLM_risks.sh \
  embed_llm_risks.py run_embed_llm_risks.sh \
  train_text_embeddings_nested_cv.py run_text_embeddings_nested_cv.sh \
  plot_text_evaluation.py text_decision_curve_analysis.py \
  run_text_posthoc_evaluation.sh

# Only if raw responses do not already exist:
./run_generate_LLM_risks.sh

# Endpoint-specific frozen embeddings:
./run_embed_llm_risks.sh

# Independent endpoint-specific nested CV:
./run_text_embeddings_nested_cv.sh

# ROC/PR, calibration, and exploratory decision curves:
./run_text_posthoc_evaluation.sh

Tmux sessions:

embed_4year_text_v3_endpoint_specific
text_nested_4year_v3_endpoint_specific
text_posthoc_4year_v3_endpoint_specific

Completion checks

ROOT=/home/sswee/music/text_nested_4year_v3_endpoint_specific

for FILE in \
  combined_evaluation/combined_evaluation_manifest.json \
  combined_evaluation/selected_primary_configuration_by_outer_fold.csv \
  combined_evaluation/all_arms_pooled_predictions_calibrated_and_classified.csv \
  combined_evaluation/pooled_performance_calibration_with_95ci.csv \
  combined_evaluation/paired_text_model_comparisons_with_95ci.csv \
  combined_evaluation/pooled_threshold_metrics_with_95ci.csv \
  combined_evaluation/pooled_confusion_matrix_counts.csv \
  text_evaluation_figures_endpoint_specific/text_roc_pr_curves.png \
  text_evaluation_figures_endpoint_specific/text_calibration_plots.png \
  text_decision_curve_analysis_endpoint_specific/text_decision_curve_summary.csv
do
  [[ -s "${ROOT}/${FILE}" ]] && echo "COMPLETE: ${FILE}" || echo "MISSING: ${FILE}"
done

Expected eligible rows per arm are 648 for SCD and 659 for PFD. Both endpoints
must contain 9 arms, with no duplicate patient-arm records.

Packaging

cd /home/sswee/music

tar -czf endpoint_specific_text_v3_results.tar.gz \
  text_nested_4year_v3_endpoint_specific/combined_evaluation \
  text_nested_4year_v3_endpoint_specific/tasks/scd/analysis_setup \
  text_nested_4year_v3_endpoint_specific/tasks/scd/selection \
  text_nested_4year_v3_endpoint_specific/tasks/pfd/analysis_setup \
  text_nested_4year_v3_endpoint_specific/tasks/pfd/selection \
  text_embeddings_4year_v3_endpoint_specific/run_metadata

tar -czf endpoint_specific_text_v3_posthoc_results.tar.gz \
  text_nested_4year_v3_endpoint_specific/text_evaluation_figures_endpoint_specific \
  text_nested_4year_v3_endpoint_specific/text_decision_curve_analysis_endpoint_specific

sha256sum \
  endpoint_specific_text_v3_results.tar.gz \
  endpoint_specific_text_v3_posthoc_results.tar.gz

Interpretation guardrails

ROC-AUC and PR-AUC use uncalibrated outer-test scores; Brier and calibration
analyses use fold-specific calibrated probabilities.

Threshold metrics use thresholds selected solely from inner out-of-fold data.

Decision curves are exploratory over 2%-25%; no clinical intervention or
independently validated action threshold was prespecified.

LLM text should not be described as adding reasoning value when the full
response does not outperform simpler representations.

Absence of significance is not evidence of equivalence.

Final highlighted-text maps are technical multimodal model-behavior
diagnostics, not attention explanations or clinically validated rationales.