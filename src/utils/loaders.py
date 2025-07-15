import json
import clip
import torch.nn as nn


def load_pairs(file_path):
    """
    Load PDB x UPKB pairs
    """
    pairs = []
    with open(file_path, "r") as f:
        for line in f:
            pair = json.loads(line.strip())
            pairs.append((pair["upkb_ac"], pair["pdb_id"]))
    return pairs


def create_clip_model_with_random_weights(model_name="ViT-B/32", device="cpu"):
    """
    Initialize a CLIP model with random weights
    """
    model, _ = clip.load(model_name, device=device, jit=False)

    def initialize_weights(module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            elif hasattr(module, "reset_parameters"):
                module.reset_parameters()

    model.apply(initialize_weights)
    return model
