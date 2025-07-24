# Data preparation and preprocessing

This section describes how to prepare the data for use with the CLASP framework, including embeddings for amino acid sequences, natural language descriptions, and PDB structures. All datasets, embeddings, and models used in CLASP are available for download from our [Internet Archive](https://archive.org/details/clasp_data).

## Table of contents

1. [Data download script](#data-download-script)
2. [Amino acid sequence embeddings](#amino-acid-sequence-embeddings)
3. [Language description embeddings](#language-description-embeddings)
4. [PDB data](#pdb-data)
    - [Option 1: Download our preprocessed PDB data](#option-1-download-preprocessed-pdb-data)
    - [Option 2: Preprocess your own PDB data](#option-2-preprocess-your-own-pdb-data)
5. [Data splitting and pairing](#data-splitting-and-pairing)
    - [Using our precomputed splits and pairs](#using-our-precomputed-splits-and-pairs)
    - [Creating your own splits and pairs](#creating-your-own-splits-and-pairs)
6. [Downloading our pre-trained models](#downloading-pre-trained-models)


## Data download script

To simplify setup, we provide a convenience script: `scripts/download.sh`. This script downloads all datasets, embeddings, and model checkpoints required to run CLASP.

From the root of the repository, make the script executable and run it:

```bash
chmod +x scripts/download.sh
./scripts/download.sh
```

After running the script, the following directory structure will be created:

```plaintext
clasp_data/
├── data/
│   ├── processed_dataset/
│   ├── aas_prott5_embeddings.h5
│   ├── processed_pdb_data.pt
│   ├── uniprot_desc_biogpt_embeddings.h5
│   ├── upkb_pdb_mapping.jsonl
├── final_models/
```

* The [`processed_dataset`](#using-our-precomputed-splits-and-pairs) directory contains data splits and pairings for training and evaluation.
* The [`final_models`](#downloading-our-pre-trained-models) directory contains pre-trained CLASP model weights.

Each of these directories is explained in detail in the sections below.





## Amino acid sequence embeddings

We use pre-trained amino acid sequence embeddings from ProtT5. You may use other embeddings, but they must follow the same format:

* File type: `.h5`
* Keys: UniProt accession numbers
* Values: 1024-dimensional embeddings (per protein)

> [Download ProtT5 embeddings](https://archive.org/download/clasp_data/clasp_data/data/aas_prott5_embeddings.h5)

## Language description embeddings

We use BioGPT embeddings of UniProt protein descriptions.

>  [Download our language embeddings](https://archive.org/download/clasp_data/clasp_data/data/uniprot_desc_biogpt_embeddings.h5)

You may also use your own language model embeddings. Just ensure they match the expected format:

* File type: `.h5`
* Keys: UniProt accession numbers
* Values: 1024-dimensional embeddings (per protein)

## PDB data

CLASP embeds PDB data internally using a graph-based geometric encoder. The following use cases require access to PDB structure data:

* Training the CLASP model 
* Generating structure embeddings
* Building cross-modal similarity matrices

#### Option 1: Download our preprocessed PDB data

If you wish to use our preprocessed PDB data directly:

>  [Download our preprocessed PDB data](https://archive.org/download/clasp_data/clasp_data/data/processed_pdb_data.pt)



#### Option 2: Preprocess your own PDB data

To preprocess your own PDB structure data into graph embeddings compatible with CLASP, use the `src/preprocess_pdb_graphs.py` script included in this repository.

**Step 1: Prepare the mapping file**

The script expects a `.jsonl` file where each line is a JSON object describing a protein, its UniProt accession code, and associated PDB IDs and chains. You can refer to our mapping file for the expected format, but here is an example for a single protein:

```json
{"upkb_ac": "P01426", "pdb": [{"id": "1IQ9", "chain": ["A"]}, {"id": "1NEA", "chain": ["A"]}]}
```

If you wish to use our original mapping file, you can download it from the following link:

>  [Download our mapping file](https://archive.org/download/clasp_data/clasp_data/data/upkb_pdb_mapping.jsonl)


**Step 2: Run the preprocessing script**

Basic usage:

```bash
python preprocess_pdb_graphs.py \
  --pdb_mapping_data path/to/your_mapping.jsonl \
  --output_file path/to/save/processed_pdb_data.pt
```

This will:

* Parse each protein's PDB ID and chains
* Construct atom-level graphs using [Graphein](https://graphein.ai/)
* Filter graphs by specified chain(s)
* Annotate each node with Meiler descriptors and 3D coordinates
* Compute edge distances as edge weights
* Save the result as a PyTorch `.pt` dictionary (`{ "<upkb_ac>-<pdb_id>": Data(...) }`)

Notes:

* This script uses [Graphein](https://graphein.ai/) to download and parse structures from the PDB. An internet connection is required unless using local files.
* If you already have `.pdb` files downloaded, refer to the optional section below for using local files.


**Optional: Use local PDB files**

If you have `.pdb` files downloaded locally, use the `--use_pdb_dir` flag and provide the directory path:

```bash
python preprocess_pdb_graphs.py \
  --pdb_mapping_data path/to/your_mapping.jsonl \
  --output_file path/to/save/processed_pdb_data.pt \
  --pdb_data_dir /path/to/local_pdbs \
  --use_pdb_dir True
```

Note that each file in the directory must be named as `<pdb_id>.pdb` (e.g., `1IQ9.pdb` for PDB ID `1IQ9`). The script will use these files instead of downloading from the PDB.

## Data splitting and pairing

We split our data into training (80%), validation (10%), and test (10%) sets. The split is done at the protein level, ensuring that no protein appears in both training and test sets. We computed this split accross 3 random seeds.

We also preperared `(upkb_ac, pdb_id)` pairs for training, validation, and testing of structure-sequence and structure-description tasks (5 different set of pairs for training to include structural diversity). For sequence-description tasks, we simply consider the paired sequence and description embeddings of the `upkb_ac`.

### Using our precomputed splits and pairs


You can download our precomputed splits and corresponding pairs:

> [Download our data splits and pairs](https://archive.org/download/clasp_data/clasp_data/data/processed_dataset/)

You will find the following files:

```plaintext
processed_dataset/
├── seed_26855092/
  ├── pairs/
    ├── test_pairs.jsonl
    ├── train_pairs_a.jsonl
    ├── train_pairs_b.jsonl
    ├── train_pairs_c.jsonl
    ├── train_pairs_d.jsonl
    ├── train_pairs_e.jsonl
    ├── val_pairs.jsonl
  ├── split/
    ├── test_upkb_pdb.jsonl
    ├── train_upkb_pdb.jsonl
    ├── val_upkb_pdb.jsonl
├── seed_119540831/ (same structure as above)
├── seed_686579303/ (same structure as above)
```

Where `pairs` subdirectory contains the pairs for training, validation, and testing, and `split` subdirectory contains the split files for each seed.

Note: the `split` files are not needed for any task or training but are provided for reference. 

### Creating your own splits and pairs

If you wish to create your own splits, ensure that you maintain the same structure and pairing logic. Each pair file should be a `.jsonl` file with each line containing a JSON object like. You can refer to our provided pairs for the expected format, but here is an example for a training pair:

```json
{"upkb_ac": "Q9FGK0", "pdb_id": "Q9FGK0-7ARB"}
```

Where the `upkb_ac` is the key used for both sequence and description embeddings, and `pdb_id` is the key for the processed PDB data.

Futhermore, you must ensure that your data directory structure matches the expected format:

```plaintext
your_split_directory/
├── test_pairs.jsonl
├── train_pairs_a.jsonl
├── train_pairs_b.jsonl
├── train_pairs_c.jsonl
├── train_pairs_d.jsonl
├── train_pairs_e.jsonl
├── val_pairs.jsonl
```

This directory path will be used when running the CLASP training script.

## Downloading our pre-trained models

While you can train CLASP from scratch using `src/train_clasp.py` (see details for training [here](training_clasp.md)), we provide pre-trained models for convenience. You can download them here:

> [Download pre-trained CLASP models](https://archive.org/download/clasp_data/clasp_data/final_models/)

You will find the following files:

```plaintext
final_models/
├── seed_26855092/
  ├── clasp_alignment.pt
  ├── clasp_pdb_encoder.pt
├── seed_119540831/ (same structure as above)
├── seed_686579303/ (same structure as above)
```

Where `clasp_alignment.pt` is the trained alignment model and `clasp_pdb_encoder.pt` is the trained PDB encoder model for each seed.

