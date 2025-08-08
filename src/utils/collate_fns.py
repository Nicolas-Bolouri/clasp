import torch
from torch_geometric.data import Batch


def pair_3modal_collate_fn(batch):
    """
    Collate function for the tri-modal dataset.
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    amino_acid_embeddings, desc_embeddings, pdb_data_list = zip(*batch)

    amino_acid_embeddings = torch.stack(amino_acid_embeddings)
    desc_embeddings = torch.stack(desc_embeddings)

    pdb_batch = Batch.from_data_list(pdb_data_list)

    return amino_acid_embeddings, desc_embeddings, pdb_batch


def pair_eval_collate_fn(batch):
    """
    Collate function for the evaluation datasets.
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    text, struct, labels = zip(*batch)
    return (
        torch.stack(text),
        torch.stack(struct),
        torch.tensor(labels, dtype=torch.long),
    )
