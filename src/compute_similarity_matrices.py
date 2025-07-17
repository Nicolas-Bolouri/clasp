import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import torch
from torch import Tensor

from models import CLASPAlignment, CLASPEncoder
from utils import create_clip_model_with_random_weights


def generate_similarity_matrices(
    aas_embeddings: Dict[str, Any],
    desc_embeddings: Dict[str, Any],
    pdb_data: Dict[str, Tensor],
    pdb_encoder: CLASPEncoder,
    alignment_model: CLASPAlignment,
    ordered_pdb_ids: List[str],
    ordered_aas_ids: List[str],
    ordered_desc_ids: List[str],
    structure_to_sequence_matrix: bool,
    structure_to_description_matrix: bool,
    sequence_to_description_matrix: bool,
    output_dir: Path,
    device: torch.device,
) -> None:
    """
    Project PDB graphs, amino‑acid embeddings, and description embeddings into
    a shared space and save their pairwise similarity matrices.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    def _collapse(proj_list: List[Tensor], name: str) -> Optional[Tensor]:
        """
        Stack a list of 1D tensors into an (N, D) tensor, or return None if empty.
        """
        if not proj_list:
            return None
        rows = []
        for t in proj_list:
            t = torch.as_tensor(t, dtype=torch.float32, device=device).squeeze()
            if t.dim() != 1:
                raise RuntimeError(
                    f"{name}: expected 1D after squeeze, got {tuple(t.shape)}"
                )
            rows.append(t.unsqueeze(0))
        return torch.cat(rows, dim=0)

    with torch.no_grad():
        # project PDBs
        pdb_proj_list: List[Tensor] = []
        for pdb_id in ordered_pdb_ids:
            graph = pdb_data.get(pdb_id)
            if graph is None:
                raise KeyError(f"PDB data missing for id: {pdb_id}")
            graph = graph.to(device)
            emb = pdb_encoder(graph)
            emb_t = torch.tensor(emb, dtype=torch.float32, device=device)
            proj = alignment_model.get_pdb_projection(emb_t)
            pdb_proj_list.append(proj)

        # project amino acids
        aas_proj_list: List[Tensor] = []
        for aas_id in ordered_aas_ids:
            raw = aas_embeddings.get(aas_id)
            if raw is None:
                raise KeyError(f"Amino acid embedding missing for id: {aas_id}")
            t = torch.tensor(raw, dtype=torch.float32, device=device)
            aas_proj_list.append(alignment_model.get_aas_projection(t))

        # project descriptions
        desc_proj_list: List[Tensor] = []
        for desc_id in ordered_desc_ids:
            raw = desc_embeddings.get(desc_id)
            if raw is None:
                raise KeyError(f"Description embedding missing for id: {desc_id}")
            t = torch.tensor(raw, dtype=torch.float32, device=device)
            desc_proj_list.append(alignment_model.get_desc_projection(t))

        pdb_proj = _collapse(pdb_proj_list, "pdb_proj")
        torch.save(pdb_proj.cpu(), output_dir / "pdb_projection.pt")
        aas_proj = _collapse(aas_proj_list, "aas_proj")
        torch.save(aas_proj.cpu(), output_dir / "aas_projection.pt")
        desc_proj = _collapse(desc_proj_list, "desc_proj")
        torch.save(desc_proj.cpu(), output_dir / "desc_projection.pt")

        print(
            f"Saved projections:"
            f" pdb={pdb_proj.shape if pdb_proj is not None else 'None'},"
            f" aas={aas_proj.shape if aas_proj is not None else 'None'},"
            f" desc={desc_proj.shape if desc_proj is not None else 'None'}"
        )

        # similarity matrices
        if (
            structure_to_sequence_matrix
            and pdb_proj is not None
            and aas_proj is not None
        ):
            sim = pdb_proj @ aas_proj.T
            torch.save(sim.cpu(), output_dir / "structure_to_sequence.pt")
            print(f"Saved structure_to_sequence.pt ({sim.shape})")

        if (
            structure_to_description_matrix
            and pdb_proj is not None
            and desc_proj is not None
        ):
            sim = pdb_proj @ desc_proj.T
            torch.save(sim.cpu(), output_dir / "structure_to_description.pt")
            print(f"Saved structure_to_description.pt ({sim.shape})")

        if (
            sequence_to_description_matrix
            and aas_proj is not None
            and desc_proj is not None
        ):
            sim = aas_proj @ desc_proj.T
            torch.save(sim.cpu(), output_dir / "sequence_to_description.pt")
            print(f"Saved sequence_to_description.pt ({sim.shape})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CLASP similarity matrices for structure, sequence, and description"
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
        help="Path to saved CLASPEncoder state_dict",
    )
    parser.add_argument(
        "--alignment_model_path",
        type=Path,
        required=True,
        help="Path to saved CLASPAlignment state_dict",
    )
    parser.add_argument(
        "--target_file",
        type=Path,
        required=True,
        help="JSON file listing pdb_ids, aas_ids, desc_ids",
    )
    parser.add_argument(
        "--structure_to_sequence_matrix",
        type=bool,
        default=True,
        help="Whether to compute structure-to-sequence similarities",
    )
    parser.add_argument(
        "--structure_to_description_matrix",
        type=bool,
        default=True,
        help="Whether to compute structure-to-description similarities",
    )
    parser.add_argument(
        "--sequence_to_description_matrix",
        type=bool,
        default=True,
        help="Whether to compute sequence-to-description similarities",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output"),
        help="Directory to save similarity matrices",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Compute device",
    )

    args = parser.parse_args()

    # validate inputs
    for p in (
        args.aas_embeddings_file,
        args.desc_embeddings_file,
        args.preprocessed_pdb_file,
        args.encoder_model_path,
        args.alignment_model_path,
        args.target_file,
    ):
        if not p.exists():
            parser.error(f"File not found: {p}")

    device = torch.device(args.device)

    # load embeddings
    with h5py.File(args.aas_embeddings_file, "r") as f:
        print("Loading amino acid embeddings...")
        aas_embeddings = {k: f[k][()] for k in f.keys()}
    with h5py.File(args.desc_embeddings_file, "r") as f:
        print("Loading description embeddings...")
        desc_embeddings = {k: f[k][()] for k in f.keys()}

    # load PDB data and targets
    print("Loading preprocessed PDB data...")
    pdb_data = torch.load(str(args.preprocessed_pdb_file), weights_only=False)

    target = json.loads(args.target_file.read_text())
    ordered_pdb_ids: List[str] = target.get("pdb_ids", [])
    ordered_aas_ids: List[str] = target.get("aas_ids", [])
    ordered_desc_ids: List[str] = target.get("desc_ids", [])

    # load models
    encoder = CLASPEncoder(
        in_channels=7,
        hidden_channels=16,
        final_embedding_size=512,
        target_size=512,
    ).to(device)
    encoder.load_state_dict(torch.load(args.encoder_model_path, map_location=device))
    encoder.eval()

    clip_base = create_clip_model_with_random_weights("ViT-B/32", device=device)
    alignment = CLASPAlignment(clip_base, embed_dim=512).to(device)
    alignment.load_state_dict(
        torch.load(args.alignment_model_path, map_location=device)
    )
    alignment.eval()

    # generate and save similarity matrices
    print("Generating similarity matrices...")
    generate_similarity_matrices(
        aas_embeddings=aas_embeddings,
        desc_embeddings=desc_embeddings,
        pdb_data=pdb_data,
        pdb_encoder=encoder,
        alignment_model=alignment,
        ordered_pdb_ids=ordered_pdb_ids,
        ordered_aas_ids=ordered_aas_ids,
        ordered_desc_ids=ordered_desc_ids,
        structure_to_sequence_matrix=args.structure_to_sequence_matrix,
        structure_to_description_matrix=args.structure_to_description_matrix,
        sequence_to_description_matrix=args.sequence_to_description_matrix,
        output_dir=args.output_dir,
        device=device,
    )


if __name__ == "__main__":
    main()
