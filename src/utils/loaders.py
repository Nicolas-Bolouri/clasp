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
