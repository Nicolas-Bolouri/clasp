# Zero-Shot Classification with CLASP

This document explains how to evaluate a trained CLASP model on three zero-shot binary classification tasks using `eval_zero_shot_classification.py`. The script computes similarity scores between modalities, selects an operating threshold on a validation set (by max F1), and reports standard metrics on the test set.

## 1. Prerequisites

Ensure you have the following input files and model components prepared. See [preprocessing documentation](data_preparation.md) for details on obtaining or generating these resources.

### Required Inputs

| Type                            | Format | Description                                                                                                               |
| ------------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------- |
| Amino-acid embeddings           | `.h5`  | HDF5 file with UniProt accessions as keys and fixed-dim vectors as values (e.g., ProtT5 output).                          |
| Description embeddings          | `.h5`  | HDF5 file with UniProt accessions as keys and fixed-dim vectors as values (e.g., BioGPT output).                          |
| Preprocessed PDB structure data | `.pt`  | Pickled Torch object mapping `<upkb_ac>-<pdb_id>` keys to PyG `Data` objects (or equivalent tensors) used by the encoder. |
| Trained CLASPEncoder model   | `.pt`  | `state_dict` of the encoder                                   |
| Trained CLASPAlignment model | `.pt`  | `state_dict` of the alignment head                            |                                                          |
| Balanced labeled pairs          | dir    | Directory with JSONL files for validation and test splits (see below).                                                    |

For models, you can use the pre-trained models provided in the [data preparation section](data_preparation.md#downloading-pre-trained-models) or train your own using the training script described in [training documentation](training_clasp.md).

## 2. Tasks Evaluated

The script can evaluate any subset of the following binary classification tasks:

1. **PDB–AAS**: Does a structure (PDB) match a sequence (AAS)?
2. **PDB–DESC**: Does a structure (PDB) match a description (DESC)?
3. **AAS–DESC**: Does a sequence (AAS) match a description (DESC)?

For each task:

* The model produces similarity scores for paired items.
* An optimal threshold is chosen on the validation set by maximizing F1.
* The test set is evaluated at that threshold and standard metrics are reported: accuracy, F1 score, ROC AUC, PR AUC, and Matthews correlation coefficient (MCC).

## 3. Running the script

```bash
python eval_zero_shot_classification.py \
  --aas_embeddings_file path/to/amino_acid_embeddings.h5 \
  --desc_embeddings_file path/to/description_embeddings.h5 \
  --preprocessed_pdb_file path/to/preprocessed_pdb.pt \
  --encoder_model_path path/to/clasp_pdb_encoder.pt \
  --alignment_model_path path/to/clasp_alignment.pt \
  --balanced_pairs_dir path/to/balanced_pairs \
  --pdb_aas True \
  --pdb_desc True \
  --aas_desc True \
  --device cuda
```

### Arguments

| Argument                  | Required | Description                                                               |
| ------------------------- | -------- | ------------------------------------------------------------------------- |
| `--aas_embeddings_file`   | ✔        | Path to `.h5` file of amino-acid sequence embeddings.                     |
| `--desc_embeddings_file`  | ✔        | Path to `.h5` file of description embeddings.                             |
| `--preprocessed_pdb_file` | ✔        | Path to `.pt` file of preprocessed PDB graphs.                            |
| `--encoder_model_path`    | ✔        | Path to a saved `CLASPEncoder` `state_dict`.                              |
| `--alignment_model_path`  | ✔        | Path to a saved `CLASPAlignment` `state_dict`.                            |
| `--balanced_pairs_dir`    | ✔        | Directory containing labeled validation/test pairs (see structure below). |
| `--pdb_aas`               | ✖        | Whether to run the **PDB–AAS** task (default `True`).                     |
| `--pdb_desc`              | ✖        | Whether to run the **PDB–DESC** task (default `True`).                    |
| `--aas_desc`              | ✖        | Whether to run the **AAS–DESC** task (default `True`).                    |
| `--device`                | ✖        | Device to use for training (`cpu` or `cuda`, default: auto-detect)     |


## 4. Balanced pairs directory

`--balanced_pairs_dir` must contain:

```
balanced_pairs/
├── aas_desc_val_pairs.jsonl
├── aas_desc_test_pairs.jsonl
├── pdb_val_pairs.jsonl
└── pdb_test_pairs.jsonl
```

Each `.jsonl` file contains one labeled pair per line, for example:

```json
[["Q8CTG7", "Q8CTG7-3BWV"], 1] // Positive pair 
[["P30131", "Q9BXW6-5ZM6"], 0] // Negative pair 
```


## 5. Output

The script prints one block per enabled task, for example:

```
============================================================
CLASP ZERO-SHOT CLASSIFICATION RESULTS - PDB-AAS
============================================================
accuracy       : 0.9292
f1_score       : 0.9304
roc_auc        : 0.9807
pr_auc         : 0.9827
mcc            : 0.8587
```

