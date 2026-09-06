MUSIC endpoint-specific multimodal modeling

Status

This directory contains the final v3 multimodal analysis for four-year sudden
cardiac death (SCD) and pump-failure death (PFD). Core nested cross-validation,
posthoc evaluation, plotting, decision curves, and patient-text shuffling are
complete.

The endpoints are independent binary tasks:

SCD: 577 survivors plus 71 SCD events (n = 648); 82 PFD patients excluded.

PFD: 577 survivors plus 82 PFD events (n = 659); 71 SCD patients excluded.

Do not mix final v3 artifacts with the old shared-head v2 multimodal results.

Files

File

Purpose

train_multimodal_nested_cv.py

Preflight, inner tuning, fold-specific selection, final fitting, aggregation, and evaluation.

run_multimodal_nested_cv.sh

Distributes task-pair-fold jobs across eight GPUs in tmux.

multimodal_posthoc_evaluation.py

Calibration, threshold metrics, paired selected-fusion comparisons, and gate diagnostics.

plot_multimodal_evaluation.py

ROC/PR, calibration, and exploratory decision-curve figures.

multimodal_text_shuffling_analysis.py

Within-outer-fold patient-text correspondence permutation test.

run_multimodal_posthoc_evaluation.sh

Runs all posthoc analyses and plots.

Multimodal_Results_Reporting.txt

Copy-ready methods, results, discussion, and reviewer-response text.

Final cross-family comparisons and attribution diagnostics are run from
7_Comparative_Analysis.

Inputs and outputs

Locked folds:
/home/sswee/music/ecg_nested_4year_three_wave/analysis_setup/nested_patient_folds.csv

Fold-specific ECG embeddings:
/home/sswee/music/ecg_nested_4year_three_wave

Endpoint-specific text embeddings:
/home/sswee/music/text_embeddings_4year_v3_endpoint_specific

Endpoint-specific text selections:
/home/sswee/music/text_nested_4year_v3_endpoint_specific

Structured clinical source:
/home/sswee/music/subject-info.csv

Output root:
/home/sswee/music/multimodal_nested_4year_v3_endpoint_specific

Locked provenance

Item

SHA-256

Locked nested folds

03190f0848904df5caadebc4ca6e3eaa64a0c409cf59f809012a3df31791cedf

Structured source

dae6975035fbc38e36bb57305021b31205f62e4e7af4b213cceeb781367fdd45

Prompt CSV

40644ed776353758c4c94bb752620f177244ac53e559d82538b7402a09f170e6

Combined pooled multimodal predictions

b9acd3437515039bab232254ecbbdc7466fc9ac2e0194473d17cbce6616a6f27

Packaged v3 multimodal results

f67b72ee03b31737dd81c12ceabd4ca6b8a7523edf317fde8fc87dfc7542174b

Verified analysis-script hashes:

2bf7941ff9b870991b2b8302075ff15b90ecb89992096e95322023f681df79fa  train_multimodal_nested_cv.py
8e2bcd0a2657b61f40302bac1bc9152363041e128bcf11f3862c6c47fa149a82  run_multimodal_nested_cv.sh
5eb4d262b0369dec3102928f2d14b52aa05a6f198b6a22281ff33749df5e758a  multimodal_posthoc_evaluation.py
e28056900e4073196b5d1bba44ffc02413a0e3576b8c9cba61770b7798d5b3c0  plot_multimodal_evaluation.py
aa6a28512161eb1ccefdfa22a236af676db79536cb9e8d3e4667c5156eee116a  multimodal_text_shuffling_analysis.py
d379cffd4237247bab6f8968cf73033a454b096156e51d56fc9f681c58eadc6d  run_multimodal_posthoc_evaluation.sh

After copying this README and reporting file, record the complete installed
directory manifest:

cd /home/sswee/multimodal_aecg_clinicalfeatures/6_Multimodal_Modeling

sha256sum \
  train_multimodal_nested_cv.py \
  run_multimodal_nested_cv.sh \
  multimodal_posthoc_evaluation.py \
  plot_multimodal_evaluation.py \
  multimodal_text_shuffling_analysis.py \
  run_multimodal_posthoc_evaluation.sh \
  README.md \
  Multimodal_Results_Reporting.txt \
  > installed_files.sha256

Prespecified modality pairs

ecg_full_text: ECG plus the endpoint-specific full LLM response.

ecg_deterministic_text: ECG plus the fixed non-LLM clinical template.

ecg_tabular: ECG plus prompt-matched structured clinical variables.

For every endpoint and pair, five fusion methods are fitted:

Direct concatenation.

Projected concatenation.

Patient-specific scalar gating.

Patient-specific vector gating.

Global weighted sum.

selected_fusion is not a sixth architecture. It concatenates the untouched
outer-test predictions from the method selected independently inside each
outer-training cohort.

Selection and leakage control

The same locked five outer folds and outer-specific inner folds are used as
in the unimodal analyses.

Competing-endpoint patients are excluded before all training operations.

Fusion architecture, capacity profile, and training duration are chosen from
the corresponding outer-training cohort only.

Configurations within 0.005 of the highest pooled inner ROC-AUC are ordered by
fewer parameters, higher PR-AUC, and configuration name.

ECG and text representations are standardized using training-split moments.

Tabular imputation, missingness indicators, scaling, and one-hot encoding are
refit inside every training split.

Platt calibration and Youden thresholds use inner out-of-fold predictions.

Every eligible patient receives one untouched prediction per endpoint, pair,
fold, and fusion arm.

Execution

cd /home/sswee/multimodal_aecg_clinicalfeatures/6_Multimodal_Modeling

chmod +x \
  train_multimodal_nested_cv.py \
  run_multimodal_nested_cv.sh \
  multimodal_posthoc_evaluation.py \
  plot_multimodal_evaluation.py \
  multimodal_text_shuffling_analysis.py \
  run_multimodal_posthoc_evaluation.sh

python -m py_compile \
  train_multimodal_nested_cv.py \
  multimodal_posthoc_evaluation.py \
  plot_multimodal_evaluation.py \
  multimodal_text_shuffling_analysis.py

bash -n run_multimodal_nested_cv.sh
bash -n run_multimodal_posthoc_evaluation.sh

./run_multimodal_nested_cv.sh

Attach to the core run:

tmux attach -t multimodal_nested_4year_v3_endpoint_specific

The core launcher schedules 30 task-pair-fold jobs per phase:

2 endpoints x 3 modality pairs x 5 outer folds = 30 jobs

After the core pipeline succeeds:

./run_multimodal_posthoc_evaluation.sh

Completion checks

ROOT=/home/sswee/music/multimodal_nested_4year_v3_endpoint_specific

echo "Task-pair-fold completion records:"
find "${ROOT}/tasks" \
  -path '*/final_models/*/outer_fold_*/run_complete.json' | wc -l

echo "Method-specific checkpoints:"
find "${ROOT}/tasks" -path '*/arms/*/checkpoint.pt' | wc -l

for FILE in \
  combined_evaluation/evaluation_manifest.json \
  combined_evaluation/all_multimodal_pooled_outer_test_predictions.csv \
  combined_evaluation/pooled_performance_with_95ci.csv \
  multimodal_posthoc_evaluation/posthoc_evaluation_manifest.json \
  multimodal_evaluation_figures/multimodal_figures_manifest.json \
  multimodal_text_shuffling/text_shuffling_manifest.json
do
  [[ -s "${ROOT}/${FILE}" ]] && echo "COMPLETE: ${FILE}" || echo "MISSING: ${FILE}"
done

Expected counts for the completed archive:

Task-pair-fold completion records: 210
Method-specific checkpoints: 150
Combined pooled prediction rows: 23,526

The 210 completion records include the staged tuning/final bookkeeping in the
archived directory; the definitive design count is 150 trained method-specific
checkpoints (2 x 3 x 5 x 5).

Posthoc analyses

The posthoc launcher produces:

selected-fusion calibration estimates and confidence intervals;

threshold metrics and confusion counts;

paired comparisons among the three selected multimodal representations;

learned gate diagnostics;

ROC and precision-recall curves;

five-group calibration plots;

exploratory 2%-25% decision curves; and

within-outer-fold text-correspondence permutation tests.

The shuffling test leaves ECGs, outcomes, fold membership, trained weights,
preprocessing, and calibration fixed and destroys only patient-text pairing.
It tests patient-specific correspondence, not whether the text is clinically
correct.

Packaging

cd /home/sswee/music

tar -czf multimodal_endpoint_specific_v3_results.tar.gz \
  multimodal_nested_4year_v3_endpoint_specific/analysis_setup \
  multimodal_nested_4year_v3_endpoint_specific/combined_selection \
  multimodal_nested_4year_v3_endpoint_specific/combined_evaluation \
  multimodal_nested_4year_v3_endpoint_specific/tasks/scd/selection \
  multimodal_nested_4year_v3_endpoint_specific/tasks/scd/pooled_outer_test \
  multimodal_nested_4year_v3_endpoint_specific/tasks/pfd/selection \
  multimodal_nested_4year_v3_endpoint_specific/tasks/pfd/pooled_outer_test \
  multimodal_nested_4year_v3_endpoint_specific/multimodal_posthoc_evaluation \
  multimodal_nested_4year_v3_endpoint_specific/multimodal_evaluation_figures \
  multimodal_nested_4year_v3_endpoint_specific/multimodal_text_shuffling

sha256sum multimodal_endpoint_specific_v3_results.tar.gz \
  > multimodal_endpoint_specific_v3_results.tar.gz.sha256

Interpretation guardrails

This is a benchmark of representations and fusion strategies, not evidence
that multimodality is universally superior.

Multimodal models must be compared formally with their component unimodal
models in 7_Comparative_Analysis; comparisons among multimodal models alone
cannot establish incremental value.

Selected architectures may differ by outer fold. Do not name a post hoc
global winning architecture.

Gate values are model diagnostics, not causal modality importance.

Decision curves are exploratory because no intervention-specific action or
validated threshold range was prespecified.

Text shuffling demonstrates patient-specific correspondence only when the
correct-pair result exceeds its within-fold permutation distribution.

Absence of a significant improvement is not proof of equivalence.