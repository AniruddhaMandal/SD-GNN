# SS-GNN/src/gxl/gxl/cli.py
import os
import json
import shutil
import argparse
import numpy as np
from pathlib import Path
from typing import List

from gxl.experiment import Experiment
from gxl.config import load_config, set_config

def _resolve_config_path(user_path: str) -> Path:
    """
    Resolve --config in three ways:
    1) If absolute or exists as given, use it.
    2) If SS_GNN_CONFIG_DIR is set, resolve relative to it.
    3) If running from repo layout, look for ../../configs relative to this file.
    """
    p = Path(user_path)
    if p.is_file():
        return p

    # Env override
    base_env = os.environ.get("SS_GNN_CONFIG_DIR")
    if base_env:
        candidate = Path(base_env) / user_path
        if candidate.is_file():
            return candidate

    # Repo layout: gxl/cli.py -> gxl/ -> src/gxl/gxl/; repo root is three levels up from 'gxl'
    repo_configs = Path(__file__).resolve().parents[3] / "configs"
    candidate = repo_configs / user_path
    if candidate.is_file():
        return candidate

    # As a last resort, try relative to CWD (typical when run from repo root)
    candidate = Path.cwd() / user_path
    if candidate.is_file():
        return candidate

    raise FileNotFoundError(
        f"Config file not found. Tried: '{user_path}', "
        f"$SS_GNN_CONFIG_DIR/{user_path}, '{repo_configs}/{user_path}', and CWD."
    )

def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """
    Overrides fields in cfg (dict) based on key=value pairs.
    Supports nested keys like 'train.lr=0.01'.
    Modifies cfg in place and also returns it.
    """
    for ov in overrides:
        if '=' not in ov:
            raise ValueError(f"Invalid override: {ov}. Must be key=value.")
        key, value = ov.split('=', 1)

        # Convert string to number/bool/null if possible
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass  # keep as string

        # Walk down nested dict keys
        parts = key.split('.')
        obj = cfg
        for p in parts[:-1]:
            if p not in obj or not isinstance(obj[p], dict):
                print(f"not present: {p}")
                obj[p] = {}  # create intermediate levels if missing
            obj = obj[p]
        obj[parts[-1]] = value
    return cfg

def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run experiment with single or multiple seeds.")
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to config file (JSON/YAML). Accepts absolute paths, or paths relative to "
             "$SS_GNN_CONFIG_DIR, or to the repo's configs/ when running from source.",
    )
    parser.add_argument(
        "--multi-seed", "-m",
        action="store_true",
        help="Run experiment with multiple seeds (default: single run).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 10, 32, 29, 75],
        help="Custom list of seeds to use when --multi-seed is set.",
    )
    parser.add_argument('--override', '-o', nargs='*', default=[],
                        help='Override config values, e.g. train.lr=0.01 model.hidden_dim=128')
    parser.add_argument('--presample', '-p',
                        action='store_true',
                        help='Presample subgraphs once before training for computational speed.')
    parser.add_argument('--name', '-n',
                        type=str,
                        default=None,
                        help='Override experiment name (default: auto-generated from model/dataset/conv)')
    args = parser.parse_args(argv)

    cfg_path = _resolve_config_path(args.config)

    cfg = load_config(str(cfg_path))

    # Apply commandline overrides
    if args.override:
        cfg = apply_overrides(cfg, args.override)

    exp_config = set_config(cfg)

    # Override name if provided via CLI
    if args.name:
        exp_config.name = args.name

    # Create timestamped run directory
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(exp_config.output_dir, exp_config.name, timestamp)
    temp_run_dir = os.path.join(".temp", run_dir)
    exp_config.experiment_dir = temp_run_dir
    exp_config.log_dir = os.path.join(temp_run_dir, "logs")
    exp_config.checkpoint_dir = os.path.join(temp_run_dir, "checkpoints")

    # Set presample flag from CLI
    if args.presample:
        exp_config.presample = True

    if args.multi_seed:
        print(f"Running experiment with multiple seeds: {args.seeds}")
        train_metrics = []
        test_metrics = []
        val_metrics = []
        for seed in args.seeds:
            # Set per-seed subdirectories
            seed_dir = os.path.join(temp_run_dir, f"seed_{seed}")
            exp_config.seed = seed
            exp_config.log_dir = os.path.join(seed_dir, "logs")
            exp_config.checkpoint_dir = os.path.join(seed_dir, "checkpoints")

            experiment = Experiment(exp_config)
            result = experiment.train()
            train_metrics.append(result['train_metric'])
            test_metrics.append(result['test_metric'])
            val_metrics.append(result['val_metric'])

            # Save per-seed result
            seed_out_str = f"Result (seed {seed}):\
                  \n\tTest: {result['test_metric']:.5f}\
                  \n\tTrain: {result['train_metric']:.5f}\
                  \n\tVal: {result['val_metric']:.5f}"
            os.makedirs(seed_dir, exist_ok=True)
            with open(os.path.join(seed_dir, "result.txt"), 'w') as f:
                f.write(seed_out_str)

        train_metrics = np.array(train_metrics)
        test_metrics = np.array(test_metrics)
        val_metrics = np.array(val_metrics)
        out_str = f"Final result over {len(args.seeds)} seeds:\
              \n\t Test: {test_metrics.mean():.5f} ± {test_metrics.std():.5f}\
              \n\t Train: {train_metrics.mean():.5f} ± {train_metrics.std():.5f}\
              \n\t Val: {val_metrics.mean():.5f} ± {val_metrics.std():.5f}\
              "
        print(out_str)

        # Save aggregated results
        os.makedirs(temp_run_dir, exist_ok=True)
        with open(os.path.join(temp_run_dir, "results.txt"), 'w') as f:
            f.write(out_str)
    else:
        print(f"Running single experiment with seed {exp_config.seed}")
        experiment = Experiment(exp_config)
        result = experiment.train()
        train_metrics = np.array(result['train_metric'])
        test_metrics = np.array(result['test_metric'])
        val_metrics = np.array(result['val_metric'])
        out_str = f"Result: \
                    \n\tTest: {test_metrics:.5f}\
                    \n\tTrain: {train_metrics:.5f}\
                    \n\tVal: {val_metrics:.5f}"
        print(out_str)

        # Save result
        os.makedirs(temp_run_dir, exist_ok=True)
        with open(os.path.join(temp_run_dir, "result.txt"), 'w') as f:
            f.write(out_str)

    # Save config.json for reproducibility
    config_save_path = os.path.join(temp_run_dir, "config.json")
    with open(config_save_path, 'w') as f:
        json.dump(cfg, f, indent=2)

    # Move completed experiment from .temp to final path
    shutil.move(temp_run_dir, run_dir)

    print(f"\nResults saved to: {run_dir}/")
    print(f"Config saved to: {os.path.join(run_dir, 'config.json')}")
