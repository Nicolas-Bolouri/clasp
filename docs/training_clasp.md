# Training CLASP

This document explains how to train the CLASP model, which aligns protein structure, sequence, and description representations using a tri-modal contrastive loss using `src/train_clasp.py`.

## 1. Prerequisites

Ensure you have the following input files and model components prepared. See [preprocessing documentation](data_preparation.md) for details on obtaining or generating these resources.

### Required Inputs

| Type                      | Format   | Description                                                                                 |
| ------------------------- | -------- | ------------------------------------------------------------------------------------------- |
| Amino acid embeddings     | `.h5`    | HDF5 file with UniProt accessions as keys and 1024D vectors as values (e.g., ProtT5 output) |
| Description embeddings    | `.h5`    | HDF5 file with UniProt accessions as keys and 1024D vectors as values (e.g., BioGPT output) |
| PDB structure graphs      | `.pt`    | Pickled Torch dictionary mapping `<upkb_ac>-<pdb_id>` to PyG `Data` objects                 |
| Training/validation pairs | `.jsonl` | Each line contains a JSON object of the form `{"upkb_ac": ..., "pdb_id": ...}`              |

## 2. Running the training script

Run the CLASP training script with the following command:

```bash
python train_clasp.py \
  --aas_embeddings_file path/to/aas_embeddings.h5 \
  --desc_embeddings_file path/to/desc_embeddings.h5 \
  --preprocessed_pdb_file path/to/preprocessed_pdb.pt \
  --processed_data_dir path/to/data_pairs \
  --checkpoint_dir path/to/checkpoints \
  --output_dir path/to/final_models \
  --seed 123 \
  --device cuda
```

### Arguments

| Argument                  | Required | Description                                                            |
| ------------------------- | -------- | ---------------------------------------------------------------------- |
| `--aas_embeddings_file`   | ✔        | Path to `.h5` file containing amino acid sequence embeddings           |
| `--desc_embeddings_file`  | ✔        | Path to `.h5` file containing protein description embeddings           |
| `--preprocessed_pdb_file` | ✔        | Path to `.pt` file containing PDB structure graphs                     |
| `--processed_data_dir`    | ✔        | Directory containing `train_pairs_*.jsonl` and `val_pairs.jsonl`       |
| `--checkpoint_dir`        | ✖        | Directory for intermediate model checkpoints (default: `checkpoints/`) |
| `--output_dir`            | ✖        | Directory to save final model checkpoints (default: `final_models/`)   |
| `--seed`                  | ✖        | Random seed (default: `42`)                                            |
| `--device`                | ✖        | Device to use for training (`cpu` or `cuda`, default: auto-detect)     |

## 3. Data directory format

The `--processed_data_dir` must contain the following files:

```plaintext
processed_data_dir/
├── train_pairs_a.jsonl
├── train_pairs_b.jsonl   
├── train_pairs_c.jsonl
├── train_pairs_d.jsonl
├── train_pairs_e.jsonl
├── val_pairs.jsonl
```

Each `.jsonl` file contains one JSON object per line:

```json
{"upkb_ac": "Q9FGK0", "pdb_id": "7ARB"}
```

## 4. Output

The following files will be saved after training:

### In `--checkpoint_dir`

| Filename            | Description                                              |
| ------------------- | -------------------------------------------------------- |
| `best_alignment.pt` | Best alignment model checkpoint (lowest validation loss) |
| `best_encoder.pt`   | Best structure encoder checkpoint                        |

### In `--output_dir`

| Filename               | Description                                         |
| ---------------------- | --------------------------------------------------- |
| `clasp_alignment.pt`   | Final alignment model (loaded from best checkpoint) |
| `clasp_pdb_encoder.pt` | Final encoder model (loaded from best checkpoint)   |


## 5. Training details

* The training script runs for 500 epochs by default.
* A learning rate scheduler (`ReduceLROnPlateau`) is used based on validation loss.
* Early stopping is triggered after 40 epochs without improvement.
* The script trains on one training split per epoch (cyclical over a–e).
* Evaluation is performed after every epoch using the fixed validation set.

---

For information on how to use the trained model for inference, see the [inference documentation](inference_utilities.md).