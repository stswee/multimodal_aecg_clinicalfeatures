# Multimodal Integration of Ambulatory ECG and Clinical Features for Sudden Cardiac Death and Pump Failure Death Prediction

## 0_Preprocessing

This folder contains scripts to preprocess the ambulatory electrocardiogram (ECG) traces from the MUSIC Dataset. All scripts may be run after downloading the MUSIC dataset and have arguments modified. 

The Jupyter Notebook,

```bash
Preprocessing.ipynb
```

provides an example for how the ambulatory ECG is preprocessed. To process all ambulatory ECG traces, run the following script:

```bash
run_preprocess_all_ecgs_HRV_complete.sh
```

## 1_Segmenting

This folder contains scripts to segment the ambulatory ECGs. After modifying the arguments, run:

```bash
create_window_index_metadata.py
```

to generate window_index_metadata.csv. This csv file will contain the indices for each ECG segment/window. Then, modify the arguments in either of the following:

```bash
run_segment_HRV_complete.sh
run_segment_HRV_complete_parallel.sh
```

Both scripts will calculate HRV features for each ECG segment. The latter script utilize multiprocessing. Run only one of the two scripts. 

## 2_Labeling

This folder contains a Jupyter Notebook to preprocess the tabular data in the MUSIC dataset. Patients are split for 5-fold cross-validation. Additionally, prompts are created for each patient. 

First, modify the Path argument in the second cell. Then, run each cell in sequence

## 3_ECG_Modeling

This folder contains scripts for generating ambulatory ECG embeddings. Each bash script (titled with "wave" in its name) corresponds to tuning a hyperparameter. Modify the arguments in each bash script, then run each script in sequence:

```bash
wave1_capacity.sh
wave2_dropout.sh
wave3_lrwd.sh
```

## 4_LLM_Modeling

This folder contains scripts to prompt an LLM (LLaMA3.1-8B-Instruct and LLaMA3.2-3B-Instruct) and generate text embeddings with a biomedical text encoder (BioBERT and ClinicalBERT).

The Jupyter Notebook,

```bash
LLM_Prompting.ipynb
```

provides an example for how an LLM is used to provide a risk assessment for the patient. To generate risk assessments and corresponding text embeddings, modify the arguments and run the following two scripts in order:

```bash
run_generate_LLM_risks.sh
run_embed_llm_risks.sh
```

After the text embeddings are generated, modify the arguments in following bash scripts (titled with "wave" in their names) for hyperparameter tuning. Then, run each script in sequence:

```bash
wave1_LLM_LM_model.sh
wave2_capacity.sh
wave3_dropout.sh
wave4_lrwd.sh
```

## 5_Multimodal_Modeling

This folder contains scripts that utilizes various multimodal fusion strategies for the ECG embeddings and text embeddings. To run one multimodal fusion strategy at a time, modify the arguments and run one of the following scripts:

```bash
run_train_multimodal_concat.sh
run_train_multimodal_projectconcat.sh
run_train_multimodal_scalar_gating.sh
run_train_multimodal_vector_gating.sh
run_train_multimodal_weighted_sum.sh
```

To run all multimodal fusion strategies and perform hyperparameter tuning, modify the arguments and run each of the following scripts in sequence:

```bash
wave1_capacity.sh
wave2_dropout.sh
wave3_lrwd.sh
```