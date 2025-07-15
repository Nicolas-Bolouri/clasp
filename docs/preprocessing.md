# Data Preprocessing

This section describes how to prepare the data for use with the CLASP framework, including embeddings for amino acid sequences, natural language descriptions, and PDB structures.

---

## Use Our Script to Download Training Data

TBD


---

## Amino Acid Sequence Embeddings

We use pre-trained amino acid sequence embeddings from ProtT5. You may use other embeddings, but they must follow the same format:

* File type: `.h5`
* Keys: UniProt accession numbers
* Values: 1024-dimensional embeddings (per protein)

> [Download ProtT5 embeddings](?)

---

## Language Description Embeddings

We use BioGPT embeddings of UniProt protein descriptions.

>  [Download our language embeddings](?)

You may also use your own language model embeddings. Just ensure they match the expected format:

* File type: `.h5`
* Keys: UniProt accession numbers
* Values: 1024-dimensional embeddings (per protein)

---

## PDB Data

CLASP embeds PDB data internally using a graph-based geometric encoder. The following use cases require access to PDB structure data:

* Training the CLASP model 
* Generating structure embeddings
* Building cross-modal similarity matrices

#### Option 1: Download Preprocessed PDB Data

If you wish to use our preprocessed PDB data directly:

>  [Download our preprocessed PDB data](?)



#### Option 2: Preprocess Your Own PDB Data

To preprocess your own PDB structure data into graph embeddings compatible with CLASP, use the `pdb_preprocess.py` script included in this repository.

**Step 1: Prepare the Mapping File**

The script expects a `.jsonl` file where each line is a JSON object describing a protein, its UniProt accession code, and associated PDB IDs and chains. Example:

```json
{"upkb_ac": "P01426", "pdb": [{"id": "1IQ9", "chain": ["A"]}, {"id": "1NEA", "chain": ["A"]}]}
```

Save this file with one JSON object per line.


**Step 2: Run the Script**

Basic usage:

```bash
python pdb_preprocessing.py \
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

---

#### Optional: Use Local PDB Files

If you have `.pdb` files downloaded locally, use the `--use_pdb_dir` flag and provide the directory path:

```bash
python pdb_preprocessing.py \
  --pdb_mapping_data path/to/your_mapping.jsonl \
  --output_file path/to/save/processed_pdb_data.pt \
  --pdb_data_dir /path/to/local_pdbs \
  --use_pdb_dir True
```

Note that each file in the directory must be named as `<pdb_id>.pdb` (e.g., `1IQ9.pdb` for PDB ID `1IQ9`). The script will use these files instead of downloading from the PDB.