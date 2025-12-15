from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import pandas as pd
import json


@dataclass
class ExperimentData:
    """Container for a single experiment's data and configuration."""
    config: Optional[Dict[str, Any]]  # run_config.json contents (None if missing)
    data: pd.DataFrame                 # statistics.parquet data
    source_dir: str                    # folder name (e.g., "facebook_opt-125m_..._20241215_143052")


class ExperimentLoader:
    """
    Load experiment results from output directories.

    Usage:
        loader = ExperimentLoader()

        # Load all experiments
        experiments = loader.load_all()

        # Load filtered experiments (AND logic)
        experiments = loader.load_filtered(model="facebook/opt-125m", n=8)

        # Access data
        for exp in experiments:
            print(f"Source: {exp.source_dir}")
            print(f"Config: {exp.config}")
            print(f"Rows: {len(exp.data)}")
    """

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize loader.

        Args:
            output_dir: Path to output directory containing experiment folders.
                        Defaults to <project>/output/ (resolved from this file's location).
        """
        if output_dir is None:
            # scripts/lib/loader.py -> scripts/lib -> scripts -> project -> project/output
            output_dir = Path(__file__).parent.parent.parent / "output"
        self.output_dir = Path(output_dir)

    def _discover_experiments(self) -> List[Path]:
        """
        Find all valid experiment directories.

        A valid experiment directory contains statistics.parquet.

        Returns:
            Sorted list of experiment directory paths.
        """
        if not self.output_dir.exists():
            return []

        experiments = []
        for subdir in self.output_dir.iterdir():
            if subdir.is_dir():
                parquet_file = subdir / "statistics.parquet"
                if parquet_file.exists():
                    experiments.append(subdir)

        return sorted(experiments)  # Consistent ordering by name (includes timestamp)

    def _load_config(self, exp_dir: Path) -> Optional[Dict[str, Any]]:
        """
        Load run_config.json from experiment directory.

        Returns:
            Config dict, or None if file doesn't exist (legacy experiments).
        """
        config_file = exp_dir / "run_config.json"
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def _load_data(self, exp_dir: Path) -> pd.DataFrame:
        """Load statistics.parquet from experiment directory."""
        parquet_file = exp_dir / "statistics.parquet"
        return pd.read_parquet(parquet_file)

    def _matches_filter(self, config: Optional[Dict[str, Any]], **filters) -> bool:
        """
        Check if config matches all specified filters (AND logic).

        Args:
            config: Experiment config dict (or None for legacy experiments)
            **filters: Key-value pairs that must all match

        Returns:
            True if config matches ALL filters, False otherwise.
            Returns False if config is None and any filters specified.
        """
        if not filters:
            return True  # No filters = match everything

        if config is None:
            return False  # Can't match filters without config

        for key, value in filters.items():
            if key not in config:
                return False  # Missing key = no match
            if config[key] != value:
                return False  # Value mismatch

        return True

    def load_all(self) -> List[ExperimentData]:
        """
        Load all experiments, regardless of config.

        Includes experiments without run_config.json (legacy data).

        Returns:
            List of ExperimentData for all valid experiment directories.
        """
        experiments = []
        for exp_dir in self._discover_experiments():
            config = self._load_config(exp_dir)
            data = self._load_data(exp_dir)
            experiments.append(ExperimentData(
                config=config,
                data=data,
                source_dir=exp_dir.name
            ))
        return experiments

    def load_filtered(self, **filters) -> List[ExperimentData]:
        """
        Load experiments matching ALL specified filters.

        Uses AND logic: experiment must match every filter to be included.
        Unspecified fields accept any value.

        Args:
            **filters: Filter criteria as keyword arguments.
                       Examples: model="facebook/opt-125m", n=8, hamming="none"

        Returns:
            List of ExperimentData matching all specified filters.

        Note:
            Experiments without run_config.json are excluded when filters are specified.

        Examples:
            # Load only experiments with n=8
            loader.load_filtered(n=8)

            # Load experiments with specific model AND hamming mode
            loader.load_filtered(model="facebook/opt-125m", hamming="standard")

            # Filter by dataset (must match exact list)
            loader.load_filtered(dataset=["ChristophSchuhmann/essays-with-instructions", "default", "train", "instructions"])
        """
        experiments = []
        for exp_dir in self._discover_experiments():
            config = self._load_config(exp_dir)

            if self._matches_filter(config, **filters):
                data = self._load_data(exp_dir)
                experiments.append(ExperimentData(
                    config=config,
                    data=data,
                    source_dir=exp_dir.name
                ))

        return experiments

    def list_experiments(self) -> List[str]:
        """
        List all experiment directory names without loading data.

        Useful for quick inspection of available experiments.

        Returns:
            List of experiment folder names.
        """
        return [exp_dir.name for exp_dir in self._discover_experiments()]
