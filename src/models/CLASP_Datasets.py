from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset


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


class EvalPairDatasetPDB(Dataset):
    """
    Pair dataset for evaluating CLASP model with labeled PDB-X pairs.
    """

    def __init__(
        self,
        labeled_pairs: List[Tuple[Tuple[str, str], Any]],
        aas_or_desc_embeddings: Dict[str, Any],
        structure_encoder: Any,
        pdb_data: Dict[str, torch.Tensor],
        device: torch.device,
    ) -> None:
        self.labeled_pairs = labeled_pairs
        self.aas_or_desc_embeddings = aas_or_desc_embeddings
        self.structure_encoder = structure_encoder
        self.pdb_data = pdb_data
        self.device = device

    def __len__(self) -> int:
        return len(self.labeled_pairs)

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor, Any]]:
        (upkb_ac, pdb_id), label = self.labeled_pairs[idx]

        graph_data = self.pdb_data.get(pdb_id, None)
        raw_aas_or_desc_embedding = self.aas_or_desc_embeddings.get(upkb_ac, None)

        if raw_aas_or_desc_embedding is None or graph_data is None:
            return None

        aas_or_desc_embedding = torch.tensor(
            raw_aas_or_desc_embedding, dtype=torch.float32
        )

        with torch.no_grad():
            structure_embedding = self.structure_encoder(graph_data.to(self.device))

        return aas_or_desc_embedding.to(self.device), structure_embedding, label


class EvalPairDatasetAASxDESC(Dataset):
    """
    Pair dataset for evaluating CLASP model with labeled AAS-DESC pairs.
    """

    def __init__(
        self,
        labeled_pairs: List[Tuple[Tuple[str, str], Any]],
        amino_acid_embeddings: Dict[str, Any],
        desc_embeddings: Dict[str, Any],
        device: torch.device,
    ) -> None:
        self.labeled_pairs = labeled_pairs
        self.amino_acid_embeddings = amino_acid_embeddings
        self.desc_embeddings = desc_embeddings
        self.device = device

    def __len__(self) -> int:
        return len(self.labeled_pairs)

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor, Any]]:
        (upkb_ac_aas, upkb_ac_desc), label = self.labeled_pairs[idx]

        raw_aas_embedding = self.amino_acid_embeddings.get(upkb_ac_aas, None)
        raw_desc_embedding = self.desc_embeddings.get(upkb_ac_desc, None)

        if raw_aas_embedding is None or raw_desc_embedding is None:
            return None

        amino_ac_embedding = torch.tensor(raw_aas_embedding, dtype=torch.float32).to(
            self.device
        )
        desc_embedding = torch.tensor(raw_desc_embedding, dtype=torch.float32).to(
            self.device
        )

        return amino_ac_embedding, desc_embedding, label
