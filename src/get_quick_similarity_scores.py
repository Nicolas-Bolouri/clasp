import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import h5py
import torch
from torch import Tensor

from models import CLASPAlignment, CLASPEncoder


def compute_and_print_quick_similarity_scores(
    pdb_id: Optional[str],
    aas_id: Optional[str],
    desc_id: Optional[str],
    pdb_data: Dict[str, Tensor],
    aas_embeddings: Dict[str, Any],
    desc_embeddings: Dict[str, Any],
    pdb_encoder: CLASPEncoder,
    alignment_model: CLASPAlignment,
    device: torch.device,
) -> None:
    """
    Compute and print quick similarity scores between one PDB, one amino-acid, and one description.
    """

    def project_pdb(pid: str) -> Tensor:
        graph = pdb_data.get(pid)
        if graph is None:
            raise KeyError(f"PDB data missing for id: {pid}")
        graph = graph.to(device)
        emb = pdb_encoder(graph)
        return alignment_model.get_pdb_projection(
            torch.tensor(emb, dtype=torch.float32, device=device)
        )

    def project_aa(aid: str) -> Tensor:
        raw = aas_embeddings.get(aid)
        if raw is None:
            raise KeyError(f"Amino acid embedding missing for id: {aid}")
        return alignment_model.get_aas_projection(
            torch.tensor(raw, dtype=torch.float32, device=device)
        )

    def project_desc(did: str) -> Tensor:
        raw = desc_embeddings.get(did)
        if raw is None:
            raise KeyError(f"Description embedding missing for id: {did}")
        return alignment_model.get_desc_projection(
            torch.tensor(raw, dtype=torch.float32, device=device)
        )

    with torch.no_grad():
        pdb_proj = project_pdb(pdb_id) if pdb_id else None
        aas_proj = project_aa(aas_id) if aas_id else None
        desc_proj = project_desc(desc_id) if desc_id else None

        # PDB <> AAS
        if pdb_proj is not None and aas_proj is not None:
            score = (pdb_proj @ aas_proj).item()
            print(f"PDB–AAS similarity ({pdb_id}, {aas_id}): {score:.6f}")
        else:
            print("PDB–AAS similarity not computed (missing data).")

        # PDB <> DESC
        if pdb_proj is not None and desc_proj is not None:
            score = (pdb_proj @ desc_proj).item()
            print(f"PDB–DESC similarity ({pdb_id}, {desc_id}): {score:.6f}")
        else:
            print("PDB–DESC similarity not computed (missing data).")

        # AAS <> DESC
        if aas_proj is not None and desc_proj is not None:
            score = (aas_proj @ desc_proj).item()
            print(f"AAS–DESC similarity ({aas_id}, {desc_id}): {score:.6f}")
        else:
            print("AAS–DESC similarity not computed (missing data).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute quick CLASP similarity scores for debugging"
    )
    parser.add_argument(
        "--aas_embeddings_file",
        type=Path,
        required=True,
        help="HDF5 file with amino-acid embeddings",
    )
    parser.add_argument(
        "--desc_embeddings_file",
        type=Path,
        required=True,
        help="HDF5 file with description embeddings",
    )
    parser.add_argument(
        "--preprocessed_pdb_file",
        type=Path,
        required=True,
        help=".pt file with preprocessed PDB graphs",
    )
    parser.add_argument(
        "--encoder_model_path",
        type=Path,
        required=True,
        help="Path to CLASPEncoder state_dict",
    )
    parser.add_argument(
        "--alignment_model_path",
        type=Path,
        required=True,
        help="Path to CLASPAlignment state_dict",
    )
    parser.add_argument(
        "--pdb_id", type=str, default=None, help="Single PDB ID to compare"
    )
    parser.add_argument(
        "--aas_id", type=str, default=None, help="Single amino-acid ID to compare"
    )
    parser.add_argument(
        "--desc_id", type=str, default=None, help="Single description ID to compare"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device",
    )

    args = parser.parse_args()

    # validate files
    for path in (
        args.aas_embeddings_file,
        args.desc_embeddings_file,
        args.preprocessed_pdb_file,
        args.encoder_model_path,
        args.alignment_model_path,
    ):
        if not path.exists():
            parser.error(f"File not found: {path}")

    device = torch.device(args.device)

    # load embeddings
    with h5py.File(args.aas_embeddings_file, "r") as f:
        print("Loading amino acid embeddings...")
        aas_embeddings = {k: f[k][()] for k in f.keys()}
    with h5py.File(args.desc_embeddings_file, "r") as f:
        print("Loading description embeddings...")
        desc_embeddings = {k: f[k][()] for k in f.keys()}

    # load PDB data and models
    print("Loading preprocessed PDB data...")
    pdb_data = torch.load(str(args.preprocessed_pdb_file), weights_only=False)

    encoder = CLASPEncoder(
        in_channels=7,
        hidden_channels=16,
        final_embedding_size=512,
        target_size=512,
    ).to(device)
    encoder.load_state_dict(torch.load(args.encoder_model_path, map_location=device))
    encoder.eval()

    aligner = CLASPAlignment(embed_dim=512).to(device)
    aligner.load_state_dict(torch.load(args.alignment_model_path, map_location=device))
    aligner.eval()

    # compute and print quick similarity scores
    print("Computing quick similarity scores...")
    compute_and_print_quick_similarity_scores(
        args.pdb_id,
        args.aas_id,
        args.desc_id,
        pdb_data,
        aas_embeddings,
        desc_embeddings,
        encoder,
        aligner,
        device,
    )


if __name__ == "__main__":
    main()
