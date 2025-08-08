import argparse
import types
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_curve,
)
from torch.utils.data import DataLoader

from models import (
    CLASPAlignment,
    CLASPEncoder,
    EvalPairDatasetAASxDESC,
    EvalPairDatasetPDB,
)
from utils import load_labeled_pairs, pair_eval_collate_fn


def compute_similarity_scores(
    model: CLASPAlignment, test_loader: DataLoader
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute similarity scores and collect labels for all pairs in the test set.
    """
    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue

            amino_embeddings, structure_embeddings, labels = batch

            logits_per_text, _ = model(amino_embeddings, structure_embeddings)
            scores = torch.diag(logits_per_text).cpu().numpy()

            all_scores.extend(scores)
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_scores), np.array(all_labels)


def find_optimal_threshold(
    ground_truth: np.ndarray, similarity_scores: np.ndarray
) -> float:
    """
    Find the optimal threshold that maximizes F1 score.
    """
    precisions, recalls, thresholds = precision_recall_curve(
        ground_truth, similarity_scores
    )
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    optimal_threshold_idx = np.argmax(f1_scores)
    return (
        thresholds[optimal_threshold_idx]
        if optimal_threshold_idx < len(thresholds)
        else thresholds[-1]
    )


def evaluate_clasp_model(
    model_name: str,
    clasp_model: CLASPAlignment,
    test_loader: DataLoader,
    threshold: float,
) -> Dict[str, Any]:
    """
    Evaluate a CLASP model at the given threshold.
    """
    clasp_model.eval()

    test_scores, test_labels = compute_similarity_scores(clasp_model, test_loader)
    binary_predictions = (test_scores >= threshold).astype(int)

    accuracy = accuracy_score(test_labels, binary_predictions)
    f1 = f1_score(test_labels, binary_predictions)

    fpr, tpr, _ = roc_curve(test_labels, test_scores)
    roc_auc = auc(fpr, tpr)
    pr_precision, pr_recall, _ = precision_recall_curve(test_labels, test_scores)
    pr_auc = auc(pr_recall, pr_precision)

    mcc = matthews_corrcoef(test_labels, binary_predictions)

    return {
        "model_name": model_name,
        "scores": test_scores,
        "labels": test_labels,
        "accuracy": accuracy,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "mcc": mcc,
    }


def print_metrics(
    metric_names: List[str],
    metrics_by_task: Dict[str, Dict[str, List[float]]],
    task_name: str,
) -> None:
    """
    Print evaluation metrics for a given task.
    """
    print("\n" + "=" * 60)
    print("CLASP ZERO-SHOT CLASSIFICATION RESULTS - " + task_name)
    print("=" * 60)
    for m in metric_names:
        vals = np.array(metrics_by_task[task_name][m])
        print(f"{m:15s}: {vals[0]:.4f}")
    print()


def evaluate_zero_shot_classification(
    amino_acid_embeddings: Dict[str, Any],
    description_embeddings: Dict[str, Any],
    pdb_data: Dict[str, torch.Tensor],
    clasp_encoder: CLASPEncoder,
    clasp_alignment: CLASPAlignment,
    pdb_val_pairs: List[Tuple[str, str, int]],
    pdb_test_pairs: List[Tuple[str, str, int]],
    aas_desc_val_pairs: List[Tuple[str, str, int]],
    aas_desc_test_pairs: List[Tuple[str, str, int]],
    device: torch.device,
    pdb_aas: bool = True,
    pdb_desc: bool = True,
    aas_desc: bool = True,
) -> Dict[str, Dict[str, List[float]]]:
    """
    Evaluate CLASP model on three zero-shot classification tasks:
    1) PDB-AAS classification
    2) PDB-DESC classification
    3) AAS-DESC classification
    """
    metric_names = [
        "accuracy",
        "f1_score",
        "roc_auc",
        "pr_auc",
        "mcc",
    ]

    tasks = ["PDB-AAS", "PDB-DESC", "AAS-DESC"]
    metrics_by_task = {task: {m: [] for m in metric_names} for task in tasks}

    # PDB-AAS CLASSIFICATION
    if pdb_aas:
        val_loader = DataLoader(
            EvalPairDatasetPDB(
                pdb_val_pairs,
                amino_acid_embeddings,
                clasp_encoder,
                pdb_data,
                device,
            ),
            batch_size=32,
            shuffle=False,
            collate_fn=pair_eval_collate_fn,
        )
        test_loader = DataLoader(
            EvalPairDatasetPDB(
                pdb_test_pairs,
                amino_acid_embeddings,
                clasp_encoder,
                pdb_data,
                device,
            ),
            batch_size=32,
            shuffle=False,
            collate_fn=pair_eval_collate_fn,
        )

        def _forward_pdb_aas(self, text_embeds, image_embeds):
            logits_pdb_aas, logits_aas_pdb = self.get_pdb_aas_logits(
                image_embeds, text_embeds
            )
            return logits_aas_pdb, logits_pdb_aas

        clasp_alignment.forward = types.MethodType(_forward_pdb_aas, clasp_alignment)

        val_scores, val_labels = compute_similarity_scores(clasp_alignment, val_loader)
        threshold = find_optimal_threshold(val_labels, val_scores)
        pdb_aas_results = evaluate_clasp_model(
            "CLASP", clasp_alignment, test_loader, threshold
        )

        for m in metric_names:
            metrics_by_task["PDB-AAS"][m].append(pdb_aas_results[m])
        print_metrics(metric_names, metrics_by_task, "PDB-AAS")

    # PDB-DESC CLASSIFICATION
    if pdb_desc:
        val_loader = DataLoader(
            EvalPairDatasetPDB(
                pdb_val_pairs,
                description_embeddings,
                clasp_encoder,
                pdb_data,
                device,
            ),
            batch_size=32,
            shuffle=False,
            collate_fn=pair_eval_collate_fn,
        )
        test_loader = DataLoader(
            EvalPairDatasetPDB(
                pdb_test_pairs,
                description_embeddings,
                clasp_encoder,
                pdb_data,
                device,
            ),
            batch_size=32,
            shuffle=False,
            collate_fn=pair_eval_collate_fn,
        )

        def _forward_pdb_desc(self, text_embeds, image_embeds):
            logits_pdb_desc, logits_desc_pdb = self.get_pdb_desc_logits(
                image_embeds, text_embeds
            )
            return logits_desc_pdb, logits_pdb_desc

        clasp_alignment.forward = types.MethodType(_forward_pdb_desc, clasp_alignment)

        val_scores, val_labels = compute_similarity_scores(clasp_alignment, val_loader)
        threshold = find_optimal_threshold(val_labels, val_scores)
        pdb_desc_results = evaluate_clasp_model(
            "CLASP", clasp_alignment, test_loader, threshold
        )

        for m in metric_names:
            metrics_by_task["PDB-DESC"][m].append(pdb_desc_results[m])
        print_metrics(metric_names, metrics_by_task, "PDB-DESC")

    # AAS-DESC CLASSIFICATION
    if aas_desc:
        val_loader = DataLoader(
            EvalPairDatasetAASxDESC(
                aas_desc_val_pairs,
                amino_acid_embeddings,
                description_embeddings,
                device,
            ),
            batch_size=32,
            shuffle=False,
            collate_fn=pair_eval_collate_fn,
        )
        test_loader = DataLoader(
            EvalPairDatasetAASxDESC(
                aas_desc_test_pairs,
                amino_acid_embeddings,
                description_embeddings,
                device,
            ),
            batch_size=32,
            shuffle=False,
            collate_fn=pair_eval_collate_fn,
        )

        def _forward_aas_desc(self, text_embeds, image_embeds):
            logits_aas_desc, logits_desc_aas = self.get_aas_desc_logits(
                text_embeds, image_embeds
            )
            return logits_desc_aas, logits_aas_desc

        clasp_alignment.forward = types.MethodType(_forward_aas_desc, clasp_alignment)

        val_scores, val_labels = compute_similarity_scores(clasp_alignment, val_loader)
        threshold = find_optimal_threshold(val_labels, val_scores)
        aas_desc_results = evaluate_clasp_model(
            "CLASP", clasp_alignment, test_loader, threshold
        )

        for m in metric_names:
            metrics_by_task["AAS-DESC"][m].append(aas_desc_results[m])
        print_metrics(metric_names, metrics_by_task, "AAS-DESC")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate CLASP model on zero-shot classification tasks"
    )
    parser.add_argument(
        "--aas_embeddings_file",
        type=Path,
        required=True,
        help="HDF5 file with amino-acid embeddings",
    )
    parser.add_argument(
        "--desc_embeddings_file",
        type=Path,
        required=True,
        help="HDF5 file with description embeddings",
    )
    parser.add_argument(
        "--preprocessed_pdb_file",
        type=Path,
        required=True,
        help=".pt file with preprocessed PDB graphs",
    )
    parser.add_argument(
        "--encoder_model_path",
        type=Path,
        required=True,
        help="Path to saved CLASPEncoder state_dict",
    )
    parser.add_argument(
        "--alignment_model_path",
        type=Path,
        required=True,
        help="Path to saved CLASPAlignment state_dict",
    )
    parser.add_argument(
        "--balanced_pairs_dir",
        type=Path,
        required=True,
        help="Directory containing balanced pair JSONL files",
    )
    parser.add_argument(
        "--pdb_aas",
        type=bool,
        default=True,
        help="Whether to evaluate PDB-AAS classification",
    )
    parser.add_argument(
        "--pdb_desc",
        type=bool,
        default=True,
        help="Whether to evaluate PDB-DESC classification",
    )
    parser.add_argument(
        "--aas_desc",
        type=bool,
        default=True,
        help="Whether to evaluate AAS-DESC classification",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Compute device",
    )

    args = parser.parse_args()

    # validate inputs
    for p in (
        args.aas_embeddings_file,
        args.desc_embeddings_file,
        args.preprocessed_pdb_file,
        args.encoder_model_path,
        args.alignment_model_path,
        args.balanced_pairs_dir,
    ):
        if not p.exists():
            parser.error(f"Path not found: {p}")

    device = torch.device(args.device)

    # load embeddings
    with h5py.File(args.aas_embeddings_file, "r") as f:
        print("Loading amino acid embeddings...")
        aas_embeddings = {k: f[k][()] for k in f.keys()}
    with h5py.File(args.desc_embeddings_file, "r") as f:
        print("Loading description embeddings...")
        desc_embeddings = {k: f[k][()] for k in f.keys()}

    # load PDB data and targets
    print("Loading preprocessed PDB data...")
    pdb_data = torch.load(str(args.preprocessed_pdb_file), weights_only=False)

    # load models
    encoder = CLASPEncoder(
        in_channels=7,
        hidden_channels=16,
        final_embedding_size=512,
        target_size=512,
    ).to(device)
    encoder.load_state_dict(torch.load(args.encoder_model_path, map_location=device))
    encoder.eval()

    alignment = CLASPAlignment(embed_dim=512).to(device)
    alignment.load_state_dict(
        torch.load(args.alignment_model_path, map_location=device)
    )
    alignment.eval()

    # load classification pairs
    aas_desc_val_pairs_path = args.balanced_pairs_dir / "aas_desc_val_pairs.jsonl"
    aas_desc_test_pairs_path = args.balanced_pairs_dir / "aas_desc_test_pairs.jsonl"
    pdb_val_pairs_path = args.balanced_pairs_dir / "pdb_val_pairs.jsonl"
    pdb_test_pairs_path = args.balanced_pairs_dir / "pdb_test_pairs.jsonl"

    if args.pdb_aas or args.pdb_desc:
        if not pdb_val_pairs_path.exists():
            parser.error(f"Path not found: {pdb_val_pairs_path}")
        if not pdb_test_pairs_path.exists():
            parser.error(f"Path not found: {pdb_test_pairs_path}")
        pdb_val_pairs = load_labeled_pairs(pdb_val_pairs_path)
        pdb_test_pairs = load_labeled_pairs(pdb_test_pairs_path)
    else:
        pdb_val_pairs = []
        pdb_test_pairs = []

    if args.aas_desc:
        if not aas_desc_val_pairs_path.exists():
            parser.error(f"Path not found: {aas_desc_val_pairs_path}")
        if not aas_desc_test_pairs_path.exists():
            parser.error(f"Path not found: {aas_desc_test_pairs_path}")
        aas_desc_val_pairs = load_labeled_pairs(aas_desc_val_pairs_path)
        aas_desc_test_pairs = load_labeled_pairs(aas_desc_test_pairs_path)
    else:
        aas_desc_val_pairs = []
        aas_desc_test_pairs = []

    evaluate_zero_shot_classification(
        amino_acid_embeddings=aas_embeddings,
        description_embeddings=desc_embeddings,
        pdb_data=pdb_data,
        clasp_encoder=encoder,
        clasp_alignment=alignment,
        pdb_val_pairs=pdb_val_pairs,
        pdb_test_pairs=pdb_test_pairs,
        aas_desc_val_pairs=aas_desc_val_pairs,
        aas_desc_test_pairs=aas_desc_test_pairs,
        device=device,
        pdb_aas=args.pdb_aas,
        pdb_desc=args.pdb_desc,
        aas_desc=args.aas_desc,
    )


if __name__ == "__main__":
    main()
