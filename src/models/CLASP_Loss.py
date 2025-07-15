import torch
import torch.nn as nn
import torch.nn.functional as F


class CLASPLoss(nn.Module):
    def __init__(self, temperature=0.07, l2_lambda=0.01):
        super(CLASPLoss, self).__init__()
        self.temperature = temperature
        self.l2_lambda = l2_lambda

    def forward(
        self,
        logits_desc_pdb,
        logits_pdb_desc,
        logits_desc_aa,
        logits_aa_desc,
        logits_pdb_aa,
        logits_aa_pdb,
        encoder_model,
    ):
        device = logits_desc_pdb.device
        batch_size = logits_desc_pdb.size(0)
        labels = torch.arange(batch_size, dtype=torch.long, device=device)

        # L1: desc <-> pdb
        loss_desc_pdb = (
            F.cross_entropy(logits_desc_pdb / self.temperature, labels)
            + F.cross_entropy(logits_pdb_desc / self.temperature, labels)
        ) / 2.0

        # L2: desc <-> amino_acids
        loss_desc_aas = (
            F.cross_entropy(logits_desc_aa / self.temperature, labels)
            + F.cross_entropy(logits_aa_desc / self.temperature, labels)
        ) / 2.0

        # L3: pdb <-> amino_acids
        loss_pdb_aas = (
            F.cross_entropy(logits_pdb_aa / self.temperature, labels)
            + F.cross_entropy(logits_aa_pdb / self.temperature, labels)
        ) / 2.0

        # Encoder L2 Regularization
        encoder_l2_loss = sum(p.pow(2).sum() for p in encoder_model.parameters())

        # Total loss
        contrastive_loss = loss_desc_pdb + loss_desc_aas + loss_pdb_aas
        total_loss = contrastive_loss + self.l2_lambda * encoder_l2_loss

        return total_loss
