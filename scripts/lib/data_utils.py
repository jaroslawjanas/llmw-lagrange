"""
Common data loading utilities for analysis scripts.

Provides unified data loading, conflict checking, and filtering functions
shared between analyze.py and attack_simulation.py.
"""
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd

from .loader import ExperimentLoader

# Parameters that should be consistent within a model group
CRITICAL_PARAMS = ['n', 'hamming', 'max_tokens', 'bias', 'temperature',
                   'hash_window', 'correct', 'green_fraction']


def check_conflicts(experiments_by_model: Dict[str, List]) -> Dict[str, Dict[str, set]]:
    """
    Check for conflicting parameters within each model group.

    Args:
        experiments_by_model: Dict mapping model names to lists of ExperimentData

    Returns:
        dict: {model: {param: set(values)}} for parameters with multiple values
    """
    conflicts = {}

    for model, experiments in experiments_by_model.items():
        model_conflicts = {}

        for param in CRITICAL_PARAMS:
            values = set()
            for exp in experiments:
                if exp.config and param in exp.config:
                    value = exp.config[param]
                    # Convert lists to tuples for hashability
                    if isinstance(value, list):
                        value = tuple(value)
                    values.add(value)

            if len(values) > 1:
                model_conflicts[param] = values

        if model_conflicts:
            conflicts[model] = model_conflicts

    return conflicts


def apply_min_tokens_filter(df: pd.DataFrame, min_tokens: Optional[int]) -> pd.DataFrame:
    """
    Apply row-level filtering based on generated_ids_count.

    Args:
        df: DataFrame with generated_ids_count and _max_tokens columns
        min_tokens: Global threshold (if specified) or None to use per-row _max_tokens

    Returns:
        Filtered DataFrame
    """
    if min_tokens is not None:
        return df[df['generated_ids_count'] >= min_tokens].copy()
    else:
        # Use per-experiment max_tokens as threshold
        return df[df['generated_ids_count'] >= df['_max_tokens']].copy()


def load_and_prepare_experiments(
    min_tokens: Optional[int] = None,
    force: bool = False,
    input_dir: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Unified data loading for analysis scripts.

    1. Load all experiments via ExperimentLoader
    2. Group by model
    3. Check for parameter conflicts (error unless force=True)
    4. Merge DataFrames per model with _max_tokens and _source columns
    5. Apply min-tokens filter

    Args:
        min_tokens: Global min tokens threshold, or None to use per-experiment max_tokens
        force: If True, proceed despite conflicting parameters
        input_dir: Input directory - can be a single experiment folder or parent of multiple.
                   If a single experiment (contains statistics.parquet), uses its parent as base.
                   Default: auto-discover from output/
        verbose: If True, print progress information

    Returns:
        dict with keys:
            - 'base_path': Path to the base directory used for loading experiments
            - 'models': {model_name: {'df': DataFrame, 'sources': list, 'config': dict, ...}}
                - df: Merged and filtered DataFrame for this model
                - sources: List of source directory names
                - config: Representative config dict (from first experiment)
                - total_rows: Total rows before filtering
                - included_rows: Rows after filtering

    Raises:
        SystemExit: If no experiments found or conflicts detected without --force
    """
    # Determine the output directory for the loader
    loader_dir = None
    single_experiment = None
    if input_dir is not None:
        input_path = Path(input_dir)
        # Check if input_dir is a single experiment (has statistics.parquet)
        if (input_path / "statistics.parquet").exists():
            # Single experiment - use parent as base, loader will find it
            loader_dir = input_path.parent
            single_experiment = input_path.name
        else:
            # Assume it's a parent directory containing experiments
            loader_dir = input_path

    # Load experiments
    loader = ExperimentLoader(output_dir=loader_dir)
    experiments = loader.load_all()

    # If single experiment specified, filter to just that one
    if single_experiment:
        experiments = [e for e in experiments if e.source_dir == single_experiment]

    if not experiments:
        print("No experiments found.")
        sys.exit(1)

    if verbose:
        print(f"Loaded {len(experiments)} experiment(s)")

    # Group by model
    experiments_by_model = {}
    for exp in experiments:
        model = exp.config.get('model', 'unknown') if exp.config else 'unknown'
        if model not in experiments_by_model:
            experiments_by_model[model] = []
        experiments_by_model[model].append(exp)

    if verbose:
        print(f"Found {len(experiments_by_model)} model(s): {list(experiments_by_model.keys())}")

    # Check for conflicts
    conflicts = check_conflicts(experiments_by_model)
    if conflicts and not force:
        print("\nERROR: Conflicting experiment parameters detected.\n")
        for model, params in conflicts.items():
            print(f"Model '{model}':")
            for param, values in params.items():
                sorted_values = sorted(str(v) for v in values)
                print(f"  - {param}: {sorted_values}")
            print()
        print("Use --force to proceed anyway (results may not be meaningful).")
        sys.exit(1)
    elif conflicts and verbose:
        print("\nWARNING: Conflicting parameters detected, proceeding with --force.\n")

    # Use the loader's resolved output_dir for config copying
    base_path = loader.output_dir

    # Process each model group
    models = {}
    for model, model_experiments in experiments_by_model.items():
        # Merge DataFrames, adding config fields
        dfs = []
        for exp in model_experiments:
            exp_df = exp.data.copy()
            exp_df['_max_tokens'] = exp.config.get('max_tokens', 0) if exp.config else 0
            exp_df['_source'] = exp.source_dir
            dfs.append(exp_df)

        model_df = pd.concat(dfs, ignore_index=True)
        total_rows = len(model_df)

        # Apply min-tokens filter
        model_df = apply_min_tokens_filter(model_df, min_tokens)
        included_rows = len(model_df)

        if included_rows == 0:
            if verbose:
                print(f"  WARNING: Model '{model}' - all {total_rows} rows filtered out. Skipping.")
            continue

        if verbose:
            print(f"  Model '{model}': {included_rows}/{total_rows} rows "
                  f"(excluded {total_rows - included_rows})")

        # Collect unique source directories
        source_dirs = model_df['_source'].unique().tolist()

        # Get representative config (from first experiment)
        rep_config = model_experiments[0].config or {}

        models[model] = {
            'df': model_df,
            'sources': source_dirs,
            'config': rep_config,
            'total_rows': total_rows,
            'included_rows': included_rows
        }

    if not models:
        print("No data remaining after filtering.")
        sys.exit(1)

    return {
        'base_path': base_path,
        'models': models
    }
