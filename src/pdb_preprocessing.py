import os
import json
import torch
import numpy as np
from torch_geometric.data import Data
from graphein.protein.graphs import construct_graph
from graphein.protein.config import ProteinGraphConfig
import argparse


def load_pdb_data(data_path):
    """
    Load PDB data from a JSON file, including UPKB_AC, PDB ID, and chain IDs.

    Parameters:
    - data_path: str, path to the JSON file containing PDB data.

    Returns:
    - list of dicts: each dict contains "upkb_ac", "pdb_id", and "chains".
    """
    pdb_data = []
    with open(data_path, "r") as f:
        for line in f:
            entry = json.loads(line.strip())
            for pdb_entry in entry["pdb"]:
                pdb_data.append(
                    {
                        "upkb_ac": entry["upkb_ac"],
                        "pdb_id": pdb_entry["id"],
                        "chains": pdb_entry["chain"],
                    }
                )
    return pdb_data


def filter_graph(graph, chains):
    """
    Filters a graph for specific chains and residues.

    Parameters:
    - graph: NetworkX graph, the initial graph to filter.
    - chains: list of str, chain IDs to keep.

    Returns:
    - NetworkX graph: a subgraph containing only the specified chains.
    """
    filtered_nodes = []
    for chain in chains:
        for node in graph.nodes:
            if graph.nodes[node]["chain_id"] == chain:
                filtered_nodes.append(node)

    return graph.subgraph(filtered_nodes)


def pdb_to_weighted_graph(pdb_id, chains, pdb_data_dir=None, use_pdb_dir=False):
    """
    Converts a PDB ID into a weighted graph compatible with encoder models.

    Parameters:
    - pdb_id: str, the PDB ID.
    - chains: list of str, chain IDs to keep.
    - pdb_data_dir: str, directory containing PDB files (if use_pdb_dir is True).
    - use_pdb_dir: bool, whether to use local PDB files.

    Returns:
    - torch_geometric.data.Data: the weighted graph.
    """
    try:
        config = ProteinGraphConfig()

        if use_pdb_dir:
            if pdb_data_dir is None:
                raise ValueError(
                    "pdb_data_dir must be provided when use_pdb_dir is True"
                )
            pdb_file_path = os.path.join(pdb_data_dir, f"{pdb_id}.pdb")
            if not os.path.exists(pdb_file_path):
                raise FileNotFoundError(f"PDB file not found at {pdb_file_path}")
            initial_graph = construct_graph(config=config, path=pdb_file_path)
        else:
            initial_graph = construct_graph(config=config, pdb_code=pdb_id)

        graph = filter_graph(initial_graph, chains)

        graph = filter_graph(initial_graph, chains)

        node_features = []
        edge_index = []
        edge_weights = []
        node_coords = []

        node_id_to_idx = {node: idx for idx, node in enumerate(graph.nodes())}

        for _, data in graph.nodes(data=True):
            node_coords.append(data["coords"])
            meiler_descriptor = data["meiler"].values
            node_features.append(meiler_descriptor)

        if not node_coords or not node_features:
            return None

        node_coords = np.array(node_coords)

        for u, v, _ in graph.edges(data=True):
            u_idx = node_id_to_idx[u]
            v_idx = node_id_to_idx[v]

            distance = np.linalg.norm(node_coords[u_idx] - node_coords[v_idx])
            edge_index.append([u_idx, v_idx])
            edge_weights.append([distance])

        if not edge_index or not edge_weights:
            return None

        node_features = np.array(node_features)
        node_features = torch.tensor(node_features, dtype=torch.float)

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_weights = torch.tensor(edge_weights, dtype=torch.float)
        node_coords = torch.tensor(node_coords, dtype=torch.float)

        return Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_weights,
            pos=node_coords,
        )

    except Exception as e:
        print(f"Error processing PDB ID {pdb_id}: {e}")
        return None


def preprocess_and_save(pdb_data, output_file, pdb_data_dir=None, use_pdb_dir=False):
    """
    Preprocess PDB data and save as PyTorch files

    Parameters:
    - pdb_data: list of dicts, each dict contains "upkb_ac", "pdb_id", and "chains".
    - output_file: str, path to save the processed data.
    - pdb_data_dir: str, directory containing PDB files (if use_pdb_dir is True).
    - use_pdb_dir: bool, whether to use local PDB files.

    Returns:
    - None, saves processed data to output_file.
    """
    processed_data_dict = {}
    total_pdbs = len(pdb_data)
    processed_count = 0

    for idx, pdb_entry in enumerate(pdb_data):
        ukpb_ac = pdb_entry["upkb_ac"]
        pdb_id = pdb_entry["pdb_id"]
        chains = pdb_entry["chains"]
        composite_id = f"{ukpb_ac}-{pdb_id}"

        if composite_id in processed_data_dict:
            processed_count += 1
            continue

        data = pdb_to_weighted_graph(pdb_id, chains, pdb_data_dir, use_pdb_dir)
        if use_pdb_dir and data is None:
            print(
                f"WARNING: PDB ID {pdb_id} not found in directory {pdb_data_dir}. Skipping."
            )
        processed_data_dict[composite_id] = data

        processed_count += 1
        print(f"Processed {processed_count}/{total_pdbs} PDB IDs")

    torch.save(processed_data_dict, output_file)
    print(f"Final data saved to {output_file}")
    print(f"Processing complete. Total PDBs: {total_pdbs}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Train CLASP")

    # required args
    parser.add_argument("--pdb_mapping_data", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)

    # optional args
    parser.add_argument("--pdb_data_dir", type=str)
    parser.add_argument("--use_pdb_dir", type=bool, default=False)

    args = parser.parse_args()

    # check pdb_mapping_data
    if not os.path.exists(args.pdb_mapping_data):
        raise FileNotFoundError(
            f"PDB mapping data file not found: {args.pdb_mapping_data}"
        )
    try:
        pdb_data = load_pdb_data(args.pdb_mapping_data)
    except Exception as e:
        raise ValueError(f"Error loading PDB mapping data: {e}")

    # output directory
    processed_pdb_data_file = args.output_file
    if not processed_pdb_data_file.endswith(".pt"):
        raise ValueError("Output file must have a .pt extension")
    os.makedirs(os.path.dirname(processed_pdb_data_file), exist_ok=True)

    # check pdb_data_dir
    if args.use_pdb_dir:
        if args.pdb_data_dir is None:
            raise ValueError("pdb_data_dir must be provided when use_pdb_dir is True")
        if not os.path.exists(args.pdb_data_dir):
            raise FileNotFoundError(
                f"PDB data directory not found: {args.pdb_data_dir}"
            )

    preprocess_and_save(
        pdb_data, processed_pdb_data_file, args.pdb_data_dir, args.use_pdb_dir
    )
