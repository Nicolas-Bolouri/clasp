import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import h5py
import torch
from torch import Tensor
from transformers import BioGptTokenizer, BioGptModel

from models import CLASPAlignment


def embed_protein_description(description: str) -> Tensor:
    """
    Tokenize and embed a protein description using BioGPT.
    """
    tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
    model = BioGptModel.from_pretrained("microsoft/biogpt")
    inputs = tokenizer(
        description, return_tensors="pt", truncation=True, max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden_state = outputs.last_hidden_state
    embedding = last_hidden_state.mean(dim=1)

    return embedding.squeeze()


def retrieve_amino_acid_embeddings(
    aa_embeddings: Dict[str, Any],
    query_desc: str,
    alignment_model: CLASPAlignment,
    aas_universe: List[str],
    top_k: int,
    output_path: Path,
    device: torch.device,
) -> List[Dict[str, Any]]:
    """
    Given a textual protein description, rank amino acids by similarity and
    write the top_k results to a JSONL file. Returns the results list.
    """
    alignment_model.eval()

    with torch.no_grad():
        desc_emb = embed_protein_description(query_desc).to(device)
        proj_desc = alignment_model.get_desc_projection(desc_emb.unsqueeze(0))

    for aa in aas_universe:
        if aa not in aa_embeddings:
            raise KeyError(f"Amino acid '{aa}' missing in embeddings")

    batch_size = 256
    scores: List[tuple] = []

    with torch.no_grad():
        for i in range(0, len(aas_universe), batch_size):
            batch_ids = aas_universe[i : i + batch_size]
            batch_tensors = [
                torch.tensor(aa_embeddings[aid], dtype=torch.float32)
                for aid in batch_ids
            ]
            batch_tensor = torch.stack(batch_tensors, dim=0).to(device)
            proj_aas = alignment_model.get_aas_projection(batch_tensor)
            sims = (proj_aas * proj_desc).sum(dim=1).cpu().tolist()
            scores.extend(zip(sims, batch_ids))

    scores.sort(key=lambda x: x[0], reverse=True)
    top_results = scores[:top_k]

    print(f"\nTop {top_k} amino acids for the query description:\n")
    for rank, (score, aid) in enumerate(top_results, start=1):
        print(f"{rank:2d}. {aid:<15} score = {score:.4f}")

    results = [
        {"rank": rank, "amino_acid_id": aid, "score": float(score)}
        for rank, (score, aid) in enumerate(top_results, start=1)
    ]
    with output_path.open("w") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank amino acids for a protein description via CLASP alignment"
    )
    parser.add_argument(
        "--aas_embeddings_file",
        type=Path,
        required=True,
        help="HDF5 file with amino acid embeddings",
    )
    parser.add_argument(
        "--query_description_file",
        type=Path,
        required=True,
        help="Text file containing the protein description",
    )
    parser.add_argument(
        "--alignment_model_path",
        type=Path,
        required=True,
        help="Path to CLASPAlignment state_dict",
    )
    parser.add_argument(
        "--aas_universe_file",
        type=Path,
        required=True,
        help="JSON file listing amino acid IDs to consider",
    )
    parser.add_argument(
        "--return_top_k",
        type=int,
        default=10,
        help="Number of top amino acids to return",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=Path("ranked_aas.jsonl"),
        help="Output JSONL file for rankings",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device",
    )

    args = parser.parse_args()

    # validate file paths
    for path in (
        args.aas_embeddings_file,
        args.query_description_file,
        args.alignment_model_path,
        args.aas_universe_file,
    ):
        if not path.exists():
            parser.error(f"File not found: {path}")

    if not args.output_file.suffix == ".jsonl":
        parser.error("`output_file` must have a .jsonl extension")

    device = torch.device(args.device)

    # load embeddings and other data
    with h5py.File(args.aas_embeddings_file, "r") as f:
        print("Loading amino acid embeddings...")
        aa_embeddings = {k: f[k][()] for k in f.keys()}

    query_desc = args.query_description_file.read_text().strip()

    universe = json.loads(args.aas_universe_file.read_text())
    if not isinstance(universe, list):
        parser.error("Amino acid universe file must contain a JSON list")

    # load model
    align_model = CLASPAlignment(embed_dim=512).to(device)
    align_model.load_state_dict(
        torch.load(args.alignment_model_path, map_location=device)
    )
    align_model.eval()

    # perform retrieval
    print("Performing amino acid retrieval...")
    retrieve_amino_acid_embeddings(
        aa_embeddings,
        query_desc,
        align_model,
        universe,
        args.return_top_k,
        args.output_file,
        device,
    )


if __name__ == "__main__":
    main()
