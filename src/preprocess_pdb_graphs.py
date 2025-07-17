import os
import json
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import torch
from torch_geometric.data import Data
from graphein.protein.graphs import construct_graph
from graphein.protein.config import ProteinGraphConfig


def load_pdb_data(data_path: str) -> List[Dict[str, Any]]:
    """
    Load PDB metadata from a line‑delimited JSON file.

    Returns a list of dicts each with keys: "upkb_ac", "pdb_id", "chains".
    """
    results: List[Dict[str, Any]] = []
    with open(data_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            for pdb_entry in entry["pdb"]:
                results.append(
                    {
                        "upkb_ac": entry["upkb_ac"],
                        "pdb_id": pdb_entry["id"],
                        "chains": pdb_entry["chain"],
                    }
                )
    return results


def filter_graph(graph, chains: List[str]):
    """
    Return the subgraph containing only nodes whose 'chain_id' is in `chains`.
    """
    nodes_to_keep = [
        n for n, d in graph.nodes(data=True) if d.get("chain_id") in chains
    ]
    return graph.subgraph(nodes_to_keep)


def pdb_to_weighted_graph(
    pdb_id: str,
    chains: List[str],
    pdb_data_dir: Optional[str] = None,
    use_pdb_dir: bool = False,
) -> Optional[Data]:
    """
    Build a torch_geometric Data object from a PDB.

    If `use_pdb_dir` is True, expects a local `{pdb_data_dir}/{pdb_id}.pdb` file.
    """
    try:
        config = ProteinGraphConfig()
        if use_pdb_dir:
            if not pdb_data_dir:
                raise ValueError(
                    "`pdb_data_dir` is required when `use_pdb_dir` is True"
                )
            pdb_path = Path(pdb_data_dir) / f"{pdb_id}.pdb"
            if not pdb_path.exists():
                raise FileNotFoundError(f"No PDB file at {pdb_path}")
            raw_graph = construct_graph(config=config, path=str(pdb_path))
        else:
            raw_graph = construct_graph(config=config, pdb_code=pdb_id)

        graph = filter_graph(raw_graph, chains)

        # collect node features and coordinates
        node_idx = {n: i for i, n in enumerate(graph.nodes())}
        coords = []
        feats = []
        for _, data in graph.nodes(data=True):
            coords.append(data["coords"])
            feats.append(data["meiler"].values)
        if not coords or not feats:
            return None

        coords = np.vstack(coords)
        # build edges
        edges, weights = [], []
        for u, v, _ in graph.edges(data=True):
            i, j = node_idx[u], node_idx[v]
            dist = np.linalg.norm(coords[i] - coords[j])
            edges.append([i, j])
            weights.append([dist])
        if not edges:
            return None

        x = torch.tensor(np.vstack(feats), dtype=torch.float)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(weights, dtype=torch.float)
        pos = torch.tensor(coords, dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)

    except Exception as e:
        print(f"Error processing {pdb_id}: {e}")
        return None


def preprocess_and_save(
    pdb_entries: List[Dict[str, Any]],
    output_file: str,
    pdb_data_dir: Optional[str] = None,
    use_pdb_dir: bool = False,
) -> None:
    """
    Convert each PDB entry to a weighted graph and save the dict of Data objects.
    """
    out: Dict[str, Optional[Data]] = {}
    total = len(pdb_entries)

    for i, entry in enumerate(pdb_entries, start=1):
        upkb_ac = entry["upkb_ac"]
        pdb_id = entry["pdb_id"]
        chains = entry["chains"]
        key = f"{upkb_ac}-{pdb_id}"

        if key not in out:
            data = pdb_to_weighted_graph(pdb_id, chains, pdb_data_dir, use_pdb_dir)
            if use_pdb_dir and data is None:
                print(f"WARNING: skipped {pdb_id} (not found in {pdb_data_dir})")
            out[key] = data

        print(f"Processed {i}/{total}: {key}")

    torch.save(out, output_file)
    print(f"Saved {len(out)} graphs to {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PDB mappings to torch_geometric Data and save."
    )
    parser.add_argument(
        "--pdb_mapping_data",
        required=True,
        help="Line‑delimited JSON with PDB mappings",
    )
    parser.add_argument(
        "--output_file", required=True, help="Output .pt file for processed graphs"
    )
    parser.add_argument("--pdb_data_dir", help="Directory of local PDB files")
    parser.add_argument(
        "--use_pdb_dir",
        action="store_true",
        help="Load structures from local PDB directory instead of downloading",
    )
    args = parser.parse_args()

    mapping_path = Path(args.pdb_mapping_data)
    if not mapping_path.exists():
        parser.error(f"No such file: {mapping_path}")

    if args.use_pdb_dir and not args.pdb_data_dir:
        parser.error("`--pdb_data_dir` is required when `--use_pdb_dir` is set")
    if args.pdb_data_dir and not Path(args.pdb_data_dir).exists():
        parser.error(f"No such directory: {args.pdb_data_dir}")

    if not args.output_file.endswith(".pt"):
        parser.error("`output_file` must end with .pt")

    entries = load_pdb_data(str(mapping_path))
    preprocess_and_save(entries, args.output_file, args.pdb_data_dir, args.use_pdb_dir)


if __name__ == "__main__":
    main()
