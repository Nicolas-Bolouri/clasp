import torch
import torch.nn as nn
import torch.nn.functional as F


class CLASPAlignment(nn.Module):
    def __init__(self, clip_model, embed_dim=512, aas_dim=1024, desc_dim=1024):
        super(CLASPAlignment, self).__init__()
        self.clip_model = clip_model
        self.pdb_projection = nn.Linear(embed_dim, embed_dim)
        self.aas_projection = nn.Linear(aas_dim, embed_dim)
        self.desc_projection = nn.Linear(desc_dim, embed_dim)

    def forward(self, pdb_embeds, aas_embeds, desc_embeds):
        projected_pdb = F.normalize(self.pdb_projection(pdb_embeds.squeeze(1)), dim=-1)
        projected_aas = F.normalize(self.aas_projection(aas_embeds), dim=-1)
        projected_desc = F.normalize(self.desc_projection(desc_embeds), dim=-1)

        logits_pdb_aas = (projected_pdb @ projected_aas.transpose(0, 1)).float()
        logits_aas_pdb = logits_pdb_aas.T

        logits_pdb_desc = (projected_pdb @ projected_desc.transpose(0, 1)).float()
        logits_desc_pdb = logits_pdb_desc.T

        logits_aas_desc = (projected_aas @ projected_desc.transpose(0, 1)).float()
        logits_desc_aas = logits_aas_desc.T

        return (
            logits_pdb_aas,
            logits_aas_pdb,
            logits_pdb_desc,
            logits_desc_pdb,
            logits_aas_desc,
            logits_desc_aas,
        )

    def get_pdb_aas_logits(self, pdb_embeds, aas_embeds):
        projected_pdb = F.normalize(self.pdb_projection(pdb_embeds.squeeze(1)), dim=-1)
        projected_aas = F.normalize(self.aas_projection(aas_embeds), dim=-1)

        logits_pdb_aas = (projected_pdb @ projected_aas.transpose(0, 1)).float()
        logits_aas_pdb = logits_pdb_aas.T
        return logits_pdb_aas, logits_aas_pdb

    def get_pdb_desc_logits(self, pdb_embeds, desc_embeds):
        projected_pdb = F.normalize(self.pdb_projection(pdb_embeds.squeeze(1)), dim=-1)
        projected_desc = F.normalize(self.desc_projection(desc_embeds), dim=-1)

        logits_pdb_desc = (projected_pdb @ projected_desc.transpose(0, 1)).float()
        logits_desc_pdb = logits_pdb_desc.T
        return logits_pdb_desc, logits_desc_pdb

    def get_aas_desc_logits(self, aas_embeds, desc_embeds):
        projected_aas = F.normalize(self.aas_projection(aas_embeds), dim=-1)
        projected_desc = F.normalize(self.desc_projection(desc_embeds), dim=-1)

        logits_aas_desc = (projected_aas @ projected_desc.transpose(0, 1)).float()
        logits_desc_aas = logits_aas_desc.T
        return logits_aas_desc, logits_desc_aas

    def get_pdb_projection(self, pdb_embeds):
        projected_pdb = F.normalize(self.pdb_projection(pdb_embeds.squeeze(1)), dim=-1)
        return projected_pdb

    def get_aas_projection(self, aas_embeds):
        projected_aas = F.normalize(self.aas_projection(aas_embeds), dim=-1)
        return projected_aas

    def get_desc_projection(self, desc_embeds):
        projected_desc = F.normalize(self.desc_projection(desc_embeds), dim=-1)
        return projected_desc
