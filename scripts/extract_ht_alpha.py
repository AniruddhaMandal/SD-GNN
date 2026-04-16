"""
extract_ht_alpha.py — Report learned HT correction strength from checkpoints.

For the proposed model (HT-pool), the readout uses:

    w_s = softmax(-alpha * log_p_s)

where alpha is a scalar learned during training. alpha > 0 amplifies rare
subgraphs; alpha = 0 recovers uniform mean pooling.

Usage
-----
# Single checkpoint
python scripts/extract_ht_alpha.py path/to/best_model.pth

# Directory — scans recursively for all best_model.pth files
python scripts/extract_ht_alpha.py experiments/

# Specific seeds from the MolHIV HT-pool runs (paper Table 2)
python scripts/extract_ht_alpha.py \\
    "experiments/ARCH-24: molhiv gine/2026-04-06_23-09-30/checkpoints/best_model.pth" \\
    "experiments/ARCH-24: molhiv gine/2026-04-09_20-32-27/seed_43/checkpoints/best_model.pth" \\
    "experiments/ARCH-24: molhiv gine/2026-04-09_20-32-27/seed_45/checkpoints/best_model.pth" \\
    "experiments/ARCH-24: molhiv gine/2026-04-10_13-48-12/seed_46/checkpoints/best_model.pth"
"""

import argparse
import os
import sys

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_checkpoints(paths: list[str]) -> list[str]:
    """Expand directories to best_model.pth files; keep explicit file paths."""
    found = []
    for p in paths:
        if os.path.isfile(p):
            found.append(p)
        elif os.path.isdir(p):
            for root, _, files in os.walk(p):
                for f in files:
                    if f == "best_model.pth":
                        found.append(os.path.join(root, f))
        else:
            print(f"[warn] path not found: {p}", file=sys.stderr)
    return sorted(found)


def load_alpha(ckpt_path: str) -> dict:
    """
    Load a checkpoint and return the HT parameters.

    Returns dict with keys:
        alpha_pool  — ht_alpha_pool value (float or None if not present)
        alpha_inter — ht_alpha_inter value (float or None if not present)
        ht_pool     — whether use_ht_pool was True in config (bool or None)
        ht_inter    — whether use_ht_inter was True in config (bool or None)
        seed        — seed from config (int or None)
        epoch       — checkpoint epoch (int or None)
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Support both save formats
    state = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))

    result = {
        "alpha_pool":  None,
        "alpha_inter": None,
        "ht_pool":     None,
        "ht_inter":    None,
        "seed":        None,
        "epoch":       ckpt.get("epoch"),
    }

    # Extract alpha values from state dict
    for key, val in state.items():
        if "ht_alpha_pool" in key:
            result["alpha_pool"] = val.item()
        if "ht_alpha_inter" in key:
            result["alpha_inter"] = val.item()

    # Extract config metadata if saved
    cfg = ckpt.get("cfg")
    if cfg is not None:
        if isinstance(cfg, dict):
            result["seed"] = cfg.get("seed")
            kwargs = cfg.get("model_config", {}).get("kwargs", {})
            result["ht_pool"]  = kwargs.get("use_ht_pool")
            result["ht_inter"] = kwargs.get("use_ht_inter")
        else:
            # ExperimentConfig dataclass
            result["seed"] = getattr(cfg, "seed", None)
            kwargs = getattr(getattr(cfg, "model_config", None), "kwargs", {})
            result["ht_pool"]  = kwargs.get("use_ht_pool")
            result["ht_inter"] = kwargs.get("use_ht_inter")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract learned HT alpha values from ARCH-24 checkpoints."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Checkpoint files (.pth) or directories to scan recursively.",
    )
    args = parser.parse_args()

    ckpt_paths = find_checkpoints(args.paths)
    if not ckpt_paths:
        print("No best_model.pth files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(ckpt_paths)} checkpoint(s).\n")
    print(f"{'Path':<70}  {'seed':>5}  {'epoch':>5}  {'ht_pool':>7}  {'alpha_pool':>10}  {'ht_inter':>8}  {'alpha_inter':>11}")
    print("-" * 130)

    pool_alphas  = []
    inter_alphas = []

    for path in ckpt_paths:
        try:
            r = load_alpha(path)
        except Exception as e:
            print(f"[error] {path}: {e}", file=sys.stderr)
            continue

        short = path[-68:] if len(path) > 68 else path
        alpha_pool_str  = f"{r['alpha_pool']:.4f}"  if r["alpha_pool"]  is not None else "N/A"
        alpha_inter_str = f"{r['alpha_inter']:.4f}" if r["alpha_inter"] is not None else "N/A"
        ht_pool_str  = str(r["ht_pool"])  if r["ht_pool"]  is not None else "?"
        ht_inter_str = str(r["ht_inter"]) if r["ht_inter"] is not None else "?"
        seed_str  = str(r["seed"])  if r["seed"]  is not None else "?"
        epoch_str = str(r["epoch"]) if r["epoch"] is not None else "?"

        print(f"{short:<70}  {seed_str:>5}  {epoch_str:>5}  {ht_pool_str:>7}  {alpha_pool_str:>10}  {ht_inter_str:>8}  {alpha_inter_str:>11}")

        if r["ht_pool"] and r["alpha_pool"] is not None:
            pool_alphas.append(r["alpha_pool"])
        if r["ht_inter"] and r["alpha_inter"] is not None:
            inter_alphas.append(r["alpha_inter"])

    # Summary statistics
    print()
    if pool_alphas:
        print(f"HT-pool alpha  (n={len(pool_alphas)}): "
              f"mean={np.mean(pool_alphas):.4f}  std={np.std(pool_alphas):.4f}  "
              f"min={np.min(pool_alphas):.4f}  max={np.max(pool_alphas):.4f}")
        print(f"  values: {[round(a, 4) for a in pool_alphas]}")
    else:
        print("No HT-pool checkpoints with use_ht_pool=True found.")

    if inter_alphas:
        print(f"HT-inter alpha (n={len(inter_alphas)}): "
              f"mean={np.mean(inter_alphas):.4f}  std={np.std(inter_alphas):.4f}  "
              f"min={np.min(inter_alphas):.4f}  max={np.max(inter_alphas):.4f}")
        print(f"  values: {[round(a, 4) for a in inter_alphas]}")

    print()
    print("Note: alpha > 0 amplifies rare subgraphs (HT correction active).")
    print("      alpha = 0 recovers uniform mean pooling.")
    print("      alpha_inter is only meaningful when use_ht_inter=True.")


if __name__ == "__main__":
    main()
