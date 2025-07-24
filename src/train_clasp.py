import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import h5py
from torch.utils.data import Dataset, DataLoader

from models import CLASPAlignment, CLASPLoss, CLASPEncoder
from utils import (
    load_pairs,
    pair_3modal_collate_fn,
)


class TriModalDataset(Dataset):
    """
    Dataset yielding (amino_acid, description, pdb_graph) triples.
    """

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        aa_embeddings: Dict[str, Any],
        desc_embeddings: Dict[str, Any],
        pdb_data: Dict[str, torch.Tensor],
        device: torch.device,
    ) -> None:
        self.pairs = pairs
        self.aa_embeddings = aa_embeddings
        self.desc_embeddings = desc_embeddings
        self.pdb_data = pdb_data
        self.device = device

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(
        self, idx: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        upkb_ac, pdb_id = self.pairs[idx]

        aa_emb = self.aa_embeddings.get(upkb_ac)
        desc_emb = self.desc_embeddings.get(upkb_ac)
        pdb_item = self.pdb_data.get(pdb_id)

        if aa_emb is None or desc_emb is None or pdb_item is None:
            return None

        amino_acid_tensor = torch.tensor(aa_emb, dtype=torch.float32).to(self.device)
        desc_tensor = torch.tensor(desc_emb, dtype=torch.float32).to(self.device)
        pdb_tensor = pdb_item.to(self.device)

        return amino_acid_tensor, desc_tensor, pdb_tensor


def train_clasp(
    seed: int,
    aa_embeddings: Dict[str, Any],
    desc_embeddings: Dict[str, Any],
    pdb_data: Dict[str, torch.Tensor],
    train_files: List[Path],
    val_pairs: List[Tuple[str, str]],
    pdb_encoder: CLASPEncoder,
    checkpoint_dir: Path,
    final_encoder_path: Path,
    final_alignment_path: Path,
    device: torch.device,
) -> Tuple[CLASPAlignment, CLASPEncoder, List[float], List[float]]:
    """
    Train the CLASP alignment model with tri‑modal contrastive loss.
    Returns the trained alignment model, encoder, and loss histories.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    # hyperparameters
    num_epochs = 500
    batch_size = 8
    lr = 1e-3
    patience = 40

    # models
    model = CLASPAlignment(embed_dim=512).to(device)
    criterion = CLASPLoss(temperature=0.07, l2_lambda=0.01)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(pdb_encoder.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=10
    )

    # data
    val_dataset = TriModalDataset(
        val_pairs, aa_embeddings, desc_embeddings, pdb_data, device
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pair_3modal_collate_fn,
    )

    best_val_loss = float("inf")
    epochs_no_improve = 0
    train_losses, val_losses = [], []

    for epoch in range(1, num_epochs + 1):
        # training
        model.train()
        pdb_encoder.train()
        total_train_loss, train_batches = 0.0, 0

        train_path = train_files[(epoch - 1) % len(train_files)]
        train_pairs = load_pairs(str(train_path))
        train_loader = DataLoader(
            TriModalDataset(
                train_pairs, aa_embeddings, desc_embeddings, pdb_data, device
            ),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=pair_3modal_collate_fn,
        )

        for batch in train_loader:
            if not batch or len(batch[0]) == 0:
                continue
            aa_t, desc_t, pdb_t = batch
            optimizer.zero_grad()
            pdb_emb = pdb_encoder(pdb_t)
            logits = model(pdb_emb, aa_t, desc_t)
            loss = criterion(*logits, encoder_model=pdb_encoder)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_batches += 1

        avg_train = total_train_loss / train_batches if train_batches else 0.0
        train_losses.append(avg_train)
        print(f"[Seed {seed}] Epoch {epoch}/{num_epochs} — Train Loss: {avg_train:.4f}")

        # validation
        model.eval()
        pdb_encoder.eval()
        total_val_loss, val_batches = 0.0, 0

        with torch.no_grad():
            for batch in val_loader:
                if not batch or len(batch[0]) == 0:
                    continue
                aa_t, desc_t, pdb_t = batch
                pdb_emb = pdb_encoder(pdb_t)
                logits = model(pdb_emb, aa_t, desc_t)
                loss = criterion(*logits, encoder_model=pdb_encoder)
                total_val_loss += loss.item()
                val_batches += 1

        avg_val = total_val_loss / val_batches if val_batches else 0.0
        val_losses.append(avg_val)
        print(f"[Seed {seed}] Epoch {epoch}/{num_epochs} — Val Loss:   {avg_val:.4f}")

        scheduler.step(avg_val)

        # save best
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            epochs_no_improve = 0
            for mdl, name in [
                (model, "best_alignment.pt"),
                (pdb_encoder, "best_encoder.pt"),
            ]:
                path = checkpoint_dir / name
                torch.save(mdl.state_dict(), path)
            print(f"[Seed {seed}] New best model saved.")
        else:
            epochs_no_improve += 1
            print(f"[Seed {seed}] No improvement for {epochs_no_improve} epochs.")

        if epochs_no_improve >= patience:
            print(f"[Seed {seed}] Early stopping.")
            break

    # load best and save final models
    model.load_state_dict(torch.load(checkpoint_dir / "best_alignment.pt"))
    pdb_encoder.load_state_dict(torch.load(checkpoint_dir / "best_encoder.pt"))

    torch.save(model.state_dict(), final_alignment_path)
    torch.save(pdb_encoder.state_dict(), final_encoder_path)
    print(
        f"[Seed {seed}] Final models saved to {final_alignment_path} and {final_encoder_path}"
    )

    return model, pdb_encoder, train_losses, val_losses


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train 3D‑CLIP with tri‑modal contrastive learning"
    )
    parser.add_argument("--aas_embeddings_file", type=Path, required=True)
    parser.add_argument("--desc_embeddings_file", type=Path, required=True)
    parser.add_argument("--preprocessed_pdb_file", type=Path, required=True)
    parser.add_argument("--processed_data_dir", type=Path, required=True)
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--output_dir", type=Path, default=Path("final_models"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    # validate paths
    for path in (
        args.aas_embeddings_file,
        args.desc_embeddings_file,
        args.preprocessed_pdb_file,
        args.processed_data_dir,
    ):
        if not path.exists():
            parser.error(f"Path not found: {path}")

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # load embeddings
    with h5py.File(args.aas_embeddings_file, "r") as f:
        print("Loading amino acid embeddings...")
        aa_embeddings = {k: f[k][()] for k in f.keys()}
    with h5py.File(args.desc_embeddings_file, "r") as f:
        print("Loading description embeddings...")
        desc_embeddings = {k: f[k][()] for k in f.keys()}

    # load PDB data, encoder, and data
    print("Loading preprocessed PDB data...")
    pdb_data = torch.load(str(args.preprocessed_pdb_file), weights_only=False)

    train_files = [
        args.processed_data_dir / f"train_pairs_{c}.jsonl"
        for c in ["a", "b", "c", "d", "e"]
    ]
    val_file = args.processed_data_dir / "val_pairs.jsonl"
    for p in train_files + [val_file]:
        if not p.exists():
            parser.error(f"Missing file: {p}")

    device = torch.device(args.device)
    pdb_encoder = CLASPEncoder(
        in_channels=7, hidden_channels=16, final_embedding_size=512, target_size=512
    ).to(device)
    pdb_encoder.eval()

    val_pairs = load_pairs(str(val_file))

    # train the model
    print("Starting training...")
    train_clasp(
        args.seed,
        aa_embeddings,
        desc_embeddings,
        pdb_data,
        train_files,
        val_pairs,
        pdb_encoder,
        args.checkpoint_dir,
        args.output_dir / "clasp_pdb_encoder.pt",
        args.output_dir / "clasp_alignment.pt",
        device,
    )


if __name__ == "__main__":
    main()
