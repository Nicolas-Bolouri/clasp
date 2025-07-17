# CLASP: Inference utilities

This document describes how to run inference using pretrained CLASP models. The provided scripts enable computing similarity matrices, projecting embeddings, and ranking amino acids based on descriptions.

## Table of contents

1. [Prerequisites](#1-prerequisites)
2. [Computing similarity matrices and obtaining projected embeddings](#2-computing-similarity-matrices)
3. [Calculating quick pairwise similarity scores](#3-quick-pairwise-similarity-scores)
4. [Ranking amino acids by description (retrieval)](#4-ranking-amino-acids-by-description)

## 1. Prerequisites

Ensure you have the following input files and model components prepared. See [preprocessing documentation](data_preparation.md) for details on obtaining or generating these resources.

| Resource                     | Format | Description                                                   |
| ---------------------------- | ------ | ------------------------------------------------------------- |
| Amino acid embeddings        | `.h5`  | UniProt IDs mapped to vectors (e.g., ProtT5)                  |
| Description embeddings       | `.h5`  | UniProt IDs mapped to vectors (e.g., BioGPT)                  |
| Preprocessed PDB graphs      | `.pt`  | Dictionary mapping `<upkb_ac>-<pdb_id>` to PyG `Data` objects |
| Trained CLASPEncoder model   | `.pt`  | `state_dict` of the encoder                                   |
| Trained CLASPAlignment model | `.pt`  | `state_dict` of the alignment head                            |

For models, you can use the pre-trained models provided in the [data preparation section](data_preparation.md#downloading-pre-trained-models) or train your own using the training script described in [training documentation](training_clasp.md).

## 2. Computing similarity matrices and obtaining projected embeddings

**Script:** `src/compute_similarity_matrices.py`

This script computes and saves projection vectors and similarity matrices for all pairs of structure, sequence, and description.

### Command

```bash
python compute_similarity_matrices.py \
  --aas_embeddings_file path/to/aas_embeddings.h5 \
  --desc_embeddings_file path/to/desc_embeddings.h5 \
  --preprocessed_pdb_file path/to/preprocessed_pdb.pt \
  --encoder_model_path path/to/clasp_pdb_encoder.pt \
  --alignment_model_path path/to/clasp_alignment.pt \
  --target_file path/to/query.json \
  --output_dir path/to/output \
  --device cuda
```

### Input: Target File Format

```json
{
  "pdb_ids": ["P12345-7ABC", ...],
  "aas_ids": ["P12345", ...],
  "desc_ids": ["P12345", ...]
}
```

### Output Files

| Filename                      | Shape      | Description                          |
| ----------------------------- | ---------- | ------------------------------------ |
| `pdb_projection.pt`           | `(Np, D)`  | Projected PDB embeddings             |
| `aas_projection.pt`           | `(Na, D)`  | Projected sequence embeddings        |
| `desc_projection.pt`          | `(Nd, D)`  | Projected description embeddings     |
| `structure_to_sequence.pt`    | `(Np, Na)` | PDB-to-AAS similarity matrix         |
| `structure_to_description.pt` | `(Np, Nd)` | PDB-to-description similarity matrix |
| `sequence_to_description.pt`  | `(Na, Nd)` | AAS-to-description similarity matrix |

## 3. Calculating quick pairwise similarity scores

**Script:** `src/get_quick_similarity_scores.py`

Use this script to compute and print raw similarity scores between a single PDB structure, amino acid, and description. Useful for debugging or quick checks.

### Command

```bash
python get_quick_similarity_scores.py \
  --aas_embeddings_file path/to/aas_embeddings.h5 \
  --desc_embeddings_file path/to/desc_embeddings.h5 \
  --preprocessed_pdb_file path/to/preprocessed_pdb.pt \
  --encoder_model_path path/to/clasp_pdb_encoder.pt \
  --alignment_model_path path/to/clasp_alignment.pt \
  --pdb_id P12345-7ABC \
  --aas_id P12345 \
  --desc_id P12345 \
  --device cuda
```

### Output Example

```
PDB–AAS similarity (P12345-7ABC, P12345): 0.832465
PDB–DESC similarity (P12345-7ABC, P12345): 0.789321
AAS–DESC similarity (P12345, P12345): 0.866775
```

## 4. Ranking amino acids by description (retrieval)

**Script:** `src/rank_amino_acids_by_description.py`

Given a protein description (natural language), this script ranks amino acids by similarity and outputs the top-k most relevant ones.

### Command

```bash
python rank_amino_acids_by_description.py \
  --aas_embeddings_file path/to/aas_embeddings.h5 \
  --query_description_file path/to/description.txt \
  --alignment_model_path path/to/clasp_alignment.pt \
  --aas_universe_file path/to/aas_list.json \
  --return_top_k 10 \
  --output_file path/to/ranked_aas.jsonl \
  --device cuda
```

### Input: Description and Universe Files

* `description.txt` should contain a single-line natural language description.
* `aas_list.json` should be a list of UniProt IDs:

```json
["P12345", "Q67890", ...]
```

### Output: JSONL Format

Each line contains a ranked amino acid:

```json
{"rank": 1, "amino_acid_id": "P12345", "score": 0.8761}
{"rank": 2, "amino_acid_id": "Q67890", "score": 0.8523}
...
```

---

For training instructions, see [train\_clasp.md](train_clasp.md).
