import json


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


def load_labeled_pairs(path):
    """
    Load labeled pair data from JSONL file.
    """
    with path.open("r") as f:
        return [((p[0][0], p[0][1]), p[1]) for p in map(json.loads, f)]
