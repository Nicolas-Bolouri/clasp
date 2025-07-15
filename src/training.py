import os
import torch
from torch.utils.data import DataLoader
from models import CLASPAlignment, CLASPLoss, CLASPEncoder
from utils import (
    load_pairs,
    pair_3modal_collate_fn,
    create_clip_model_with_random_weights,
)
from torch.utils.data import Dataset
import random
import h5py
import argparse


class TriModalDataset(Dataset):
    def __init__(self, pairs, amino_acid_embeddings, desc_embeddings, pdb_data, device):
        self.pairs = pairs
        self.amino_acid_embeddings = amino_acid_embeddings
        self.desc_embeddings = desc_embeddings
        self.pdb_data = pdb_data
        self.device = device

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        upkb_ac, pdb_id = self.pairs[idx]

        if (
            upkb_ac not in self.amino_acid_embeddings
            or upkb_ac not in self.desc_embeddings
        ):
            return None

        amino_ac_embedding = torch.tensor(
            self.amino_acid_embeddings[upkb_ac], dtype=torch.float32
        ).to(self.device)

        desc_embedding = torch.tensor(
            self.desc_embeddings[upkb_ac], dtype=torch.float32
        ).to(self.device)

        pdb_data_item = self.pdb_data.get(pdb_id)

        if pdb_data_item is None:
            return None

        pdb_data_item = pdb_data_item.to(self.device)

        return amino_ac_embedding, desc_embedding, pdb_data_item


def train_clasp(
    seed,
    amino_acid_embeddings,
    desc_embeddings,
    pdb_data,
    train_file_paths,
    val_pairs,
    pdb_encoder,
    checkpoint_dir,
    final_encoder_path,
    final_3dclip_path,
    device,
):
    """
    Training function for 3D CLIP model with tri-modal contrastive learning for a specific seed.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    # training params
    num_epochs = 500
    batch_size = 8
    learning_rate = 0.001
    patience = 40
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    # init models
    clip_model = create_clip_model_with_random_weights("ViT-B/32", device=device)
    model_3dclip = CLASPAlignment(clip_model, embed_dim=512).to(device)
    criterion = CLASPLoss(temperature=0.07, l2_lambda=0.01)

    # optimizer and scheduler
    optimizer = torch.optim.Adam(
        list(model_3dclip.parameters()) + list(pdb_encoder.parameters()),
        lr=learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=10
    )

    # load validation data
    val_dataset = TriModalDataset(
        val_pairs, amino_acid_embeddings, desc_embeddings, pdb_data, device
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pair_3modal_collate_fn,
    )

    train_losses = []
    val_losses = []

    # train loop
    for epoch in range(num_epochs):
        model_3dclip.train()
        pdb_encoder.train()
        total_loss = 0
        num_batches = 0

        train_file_path = train_file_paths[epoch % len(train_file_paths)]
        train_pairs = load_pairs(train_file_path)
        train_dataset = TriModalDataset(
            train_pairs, amino_acid_embeddings, desc_embeddings, pdb_data, device
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=pair_3modal_collate_fn,
        )

        for batch in train_loader:
            if batch is None or len(batch) < 3 or len(batch[0]) == 0:
                continue

            amino_acid_embs, desc_embs, pdb_data_dict = batch
            optimizer.zero_grad()
            pdb_embeds = pdb_encoder(pdb_data_dict)

            (
                logits_pdb_aas,
                logits_aas_pdb,
                logits_pdb_desc,
                logits_desc_pdb,
                logits_aas_desc,
                logits_desc_aas,
            ) = model_3dclip(pdb_embeds, amino_acid_embs, desc_embs)

            loss = criterion(
                logits_desc_pdb,
                logits_pdb_desc,
                logits_desc_aas,
                logits_aas_desc,
                logits_pdb_aas,
                logits_aas_pdb,
                clip_model=model_3dclip.clip_model,
                encoder_model=pdb_encoder,
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0
        print(
            f"Seed {seed}: Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}"
        )
        train_losses.append(avg_train_loss)

        # val phase
        model_3dclip.eval()
        pdb_encoder.eval()
        val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                if batch is None or len(batch) < 3 or len(batch[0]) == 0:
                    continue

                amino_acid_embs, desc_embs, pdb_data_dict = batch
                pdb_embeds = pdb_encoder(pdb_data_dict)

                (
                    logits_pdb_aas,
                    logits_aas_pdb,
                    logits_pdb_desc,
                    logits_desc_pdb,
                    logits_aas_desc,
                    logits_desc_aas,
                ) = model_3dclip(pdb_embeds, amino_acid_embs, desc_embs)

                loss = criterion(
                    logits_desc_pdb,
                    logits_pdb_desc,
                    logits_desc_aas,
                    logits_aas_desc,
                    logits_pdb_aas,
                    logits_aas_pdb,
                    clip_model=model_3dclip.clip_model,
                    encoder_model=pdb_encoder,
                )

                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        print(
            f"Seed {seed}: Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}"
        )
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        # check for improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0

            print(f"Seed {seed}: Validation loss improved.")

            torch.save(
                model_3dclip.state_dict(),
                os.path.join(checkpoint_dir, "best_3dclip_model.pt"),
            )
            torch.save(
                pdb_encoder.state_dict(),
                os.path.join(checkpoint_dir, "best_pdb_encoder_model.pt"),
            )
            print(f"Seed {seed}: Best models saved to {checkpoint_dir}")

        else:
            epochs_without_improvement += 1
            print(
                f"Seed {seed}: No improvement in validation loss for {epochs_without_improvement} epoch(s)."
            )

        # early stopping check
        if epochs_without_improvement >= patience:
            print(f"Seed {seed}: Early stopping triggered.")
            break

    model_3dclip.load_state_dict(
        torch.load(os.path.join(checkpoint_dir, "best_3dclip_model.pt"))
    )
    pdb_encoder.load_state_dict(
        torch.load(os.path.join(checkpoint_dir, "best_pdb_encoder_model.pt"))
    )

    if final_3dclip_path and final_encoder_path:
        torch.save(model_3dclip.state_dict(), final_3dclip_path)
        torch.save(pdb_encoder.state_dict(), final_encoder_path)
        print(
            f"Seed {seed}: Final models saved to {final_3dclip_path} and {final_encoder_path}"
        )

    return model_3dclip, pdb_encoder, train_losses, val_losses


if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    parser = argparse.ArgumentParser(description="Train CLASP")

    # arg parsing
    parser.add_argument("--aas_embeddings_path", type=str, required=True)
    parser.add_argument("--desc_embeddings_path", type=str, required=True)
    parser.add_argument("--preprocessed_pdb_file", type=str, required=True)
    parser.add_argument(
        "--train_file_paths",
        type=str,
        nargs="+",
        required=True,
        help="List of training pair JSONL files",
    )
    parser.add_argument("--val_file_path", type=str, required=True)
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints/clasp_training"
    )
    parser.add_argument(
        "--final_alignment_module_path",
        type=str,
        default="final_models/clasp_alignment.pt",
    )
    parser.add_argument(
        "--final_encoder_module_path",
        type=str,
        default="final_models/clasp_pdb_encoder.pt",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    args = parser.parse_args()

    # destructure args

    # check if paths exist
    if not os.path.exists(args.aas_embeddings_path):
        raise FileNotFoundError(
            f"Amino acid embeddings file not found: {args.aas_embeddings_path}"
        )
    if not os.path.exists(args.desc_embeddings_path):
        raise FileNotFoundError(
            f"Descriptor embeddings file not found: {args.desc_embeddings_path}"
        )
    if not os.path.exists(args.preprocessed_pdb_file):
        raise FileNotFoundError(
            f"Preprocessed PDB file not found: {args.preprocessed_pdb_file}"
        )
    for train_file_path in args.train_file_paths:
        if not os.path.exists(train_file_path):
            raise FileNotFoundError(f"Training file not found: {train_file_path}")
    if not os.path.exists(args.val_file_path):
        raise FileNotFoundError(f"Validation file not found: {args.val_file_path}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.final_alignment_module_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.final_encoder_module_path), exist_ok=True)

    # ensure data is in correct format
    try:
        with h5py.File(args.aas_embeddings_path, "r") as f:
            amino_acid_embeddings = {k: f[k][()] for k in f.keys()}
    except Exception as e:
        raise ValueError(f"Error loading amino acid embeddings: {e}")

    try:
        with h5py.File(args.desc_embeddings_path, "r") as f:
            desc_embeddings = {k: f[k][()] for k in f.keys()}
    except Exception as e:
        raise ValueError(f"Error loading descriptor embeddings: {e}")

    try:
        pdb_data = torch.load(args.preprocessed_pdb_file)
    except Exception as e:
        raise ValueError(f"Error loading preprocessed PDB file: {e}")

    try:
        train_file_paths = [os.path.abspath(path) for path in args.train_file_paths]
        for train_file_path in train_file_paths:
            train_pairs = load_pairs(train_file_path)
    except Exception as e:
        raise ValueError(f"Error loading training pairs: {e}")

    try:
        val_pairs = load_pairs(args.val_file_path)
    except Exception as e:
        raise ValueError(f"Error loading validation pairs: {e}")

    # set device and seed
    if args.device not in ["cpu", "cuda"]:
        raise ValueError("Device must be 'cpu' or 'cuda'")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available on this machine, use 'cpu' instead")
    device = torch.device(args.device)
    seed = args.seed

    # init encoder
    pdb_encoder = CLASPEncoder(
        in_channels=7, hidden_channels=16, final_embedding_size=512, target_size=512
    ).to(device)
    pdb_encoder.eval()

    train_clasp(
        seed,
        amino_acid_embeddings,
        desc_embeddings,
        pdb_data,
        train_file_paths,
        val_pairs,
        pdb_encoder,
        args.checkpoint_dir,
        args.final_encoder_module_path,
        args.final_alignment_module_path,
        device,
    )
