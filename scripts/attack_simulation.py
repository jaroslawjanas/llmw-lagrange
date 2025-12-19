"""
Attack simulation script for watermark robustness testing.

Simulates insertion, deletion, and substitution attacks on watermarked text
and measures watermark recovery rate at various attack budgets.
"""
import argparse
import json
import multiprocessing as mp
import random
import shutil
import sys
import time
from datetime import datetime
from multiprocessing import cpu_count
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import galois

# Add parent directory to path for src imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from lib import load_and_prepare_experiments
from src.llm_watermark import LLMWatermarkDecoder, MCPSolver
from src.pm_galois import GaloisField


def get_device(no_cuda: bool) -> str:
    """Determine device based on availability and user preference.

    WARNING: torch.randperm produces different results on CPU vs CUDA.
    The decoder MUST use the same device as was used during encoding.
    """
    if no_cuda:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulate attacks on watermarked text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python attack_simulation.py                          # Default 10% perturbation
  python attack_simulation.py --perturbation-rate 20   # 20% max perturbation
  python attack_simulation.py --min-tokens 200         # Filter short sequences
        """
    )
    parser.add_argument("--min-tokens", type=int, default=None,
                        help="Min tokens filter. Default: use each experiment's max_tokens")
    parser.add_argument("--perturbation-rate", type=int, default=10,
                        help="Max perturbation as %% of tokens (default: 10)")
    parser.add_argument("--output-dir", type=str, default="output/attacks",
                        help="Output directory for results (default: output/attacks)")
    parser.add_argument("--force", action="store_true",
                        help="Proceed even with conflicting parameters")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--groups", type=str, default="1",
                        help="Comma-separated list of group counts to test (default: 1). "
                             "Example: '1, 2, 3, 4, 5' tests 1 through 5 groups.")
    parser.add_argument("--no-cuda", action="store_true",
                        help="Disable CUDA even if available")
    parser.add_argument("-j", "--workers", type=int, default=0,
                        help="Number of worker processes (0=auto, 1=serial, default: 0)")
    return parser.parse_args()


# =============================================================================
# Group Helpers
# =============================================================================

def parse_groups_arg(groups_arg: str) -> List[int]:
    """Parse --groups argument to a list of integers.

    Args:
        groups_arg: Comma-separated list of integers, e.g., "1, 2, 3, 4, 5"

    Returns:
        List of group counts to test

    Raises:
        ValueError: If parsing fails or values are invalid
    """
    groups = []
    for part in groups_arg.split(','):
        part = part.strip()
        if not part:
            continue
        val = int(part)
        if val < 1:
            raise ValueError(f"Group count must be >= 1, got {val}")
        groups.append(val)

    if not groups:
        raise ValueError("No valid group counts provided")

    return sorted(set(groups))  # Remove duplicates and sort


def split_budget(budget: int, num_groups: int) -> List[int]:
    """Split budget into num_groups sizes.

    Example: split_budget(10, 3) → [4, 3, 3]
    """
    if num_groups <= 0:
        return []
    num_groups = min(num_groups, budget)  # Can't have more groups than budget
    if num_groups == 0:
        return []

    base = budget // num_groups
    remainder = budget % num_groups
    return [base + 1] * remainder + [base] * (num_groups - remainder)


def distribute_gaps(free_space: int, num_gaps: int, rng: random.Random) -> List[int]:
    """Randomly distribute free_space among num_gaps bins.

    Example: distribute_gaps(10, 3, rng) might return [3, 4, 3]
    """
    if num_gaps <= 0:
        return []
    if free_space <= 0:
        return [0] * num_gaps

    gaps = [0] * num_gaps
    for _ in range(free_space):
        gaps[rng.randint(0, num_gaps - 1)] += 1
    return gaps


def compute_group_positions(text_length: int, group_sizes: List[int],
                            rng: random.Random) -> List[int]:
    """Compute non-overlapping start positions for groups.

    Returns list of start positions, one per group.
    Raises ValueError if groups don't fit.
    """
    if not group_sizes:
        return []

    total_budget = sum(group_sizes)
    free_space = text_length - total_budget

    if free_space < 0:
        raise ValueError(f"Groups don't fit: need {total_budget}, have {text_length}")

    num_gaps = len(group_sizes) + 1
    gaps = distribute_gaps(free_space, num_gaps, rng)

    positions = []
    current_pos = gaps[0]
    for i, size in enumerate(group_sizes):
        positions.append(current_pos)
        current_pos += size + gaps[i + 1]

    return positions


# =============================================================================
# Attack Functions
# =============================================================================

def insertion_attack(token_ids: List[int], group_sizes: List[int],
                     positions: List[int], vocab_size: int,
                     rng: random.Random) -> List[int]:
    """
    Insert contiguous groups of random tokens at specified positions.

    Args:
        token_ids: Original token IDs
        group_sizes: List of sizes for each group to insert
        positions: List of start positions for each group
        vocab_size: Vocabulary size for random token generation
        rng: Random number generator

    Returns:
        Modified token IDs with insertions
    """
    result = token_ids.copy()

    # Process right-to-left to maintain position validity
    for size, pos in sorted(zip(group_sizes, positions), key=lambda x: -x[1]):
        tokens_to_insert = [rng.randint(0, vocab_size - 1) for _ in range(size)]
        # Insert all tokens at position (they become contiguous)
        for i, token in enumerate(tokens_to_insert):
            result.insert(pos + i, token)

    return result


def deletion_attack(token_ids: List[int], group_sizes: List[int],
                    positions: List[int], rng: random.Random) -> List[int]:
    """
    Delete contiguous groups of tokens at specified positions.

    Args:
        token_ids: Original token IDs
        group_sizes: List of sizes for each group to delete
        positions: List of start positions for each group
        rng: Random number generator

    Returns:
        Modified token IDs with deletions
    """
    result = token_ids.copy()

    # Process right-to-left to maintain position validity
    for size, pos in sorted(zip(group_sizes, positions), key=lambda x: -x[1]):
        del result[pos:pos + size]

    return result


def substitution_attack(token_ids: List[int], group_sizes: List[int],
                        positions: List[int], vocab_size: int,
                        rng: random.Random) -> List[int]:
    """
    Replace contiguous groups of tokens with random tokens.

    Args:
        token_ids: Original token IDs
        group_sizes: List of sizes for each group to substitute
        positions: List of start positions for each group
        vocab_size: Vocabulary size for random token generation
        rng: Random number generator

    Returns:
        Modified token IDs with substitutions
    """
    result = token_ids.copy()

    for size, pos in zip(group_sizes, positions):
        for i in range(size):
            if pos + i < len(result):
                result[pos + i] = rng.randint(0, vocab_size - 1)

    return result


# =============================================================================
# Decoder Cache
# =============================================================================

class SimulationCache:
    """Cache for expensive objects: decoders, GF instances, MCP solvers."""

    def __init__(self):
        self._decoders: Dict[str, LLMWatermarkDecoder] = {}
        self._galois_gf: Dict[int, object] = {}  # {n: galois.GF(2**n)}
        self._pm_gf: Dict[int, GaloisField] = {}  # {n: GaloisField(n)}
        self._mcp_solvers: Dict[int, MCPSolver] = {}  # {n: MCPSolver}
        self.timing = {'gf_mcp': 0.0, 'decoder': 0.0, 'attack': 0.0, 'decode': 0.0, 'mcp': 0.0}

    def print_timing(self):
        """Print timing breakdown."""
        print("\nTiming breakdown:")
        total = sum(self.timing.values())
        for name, t in sorted(self.timing.items(), key=lambda x: -x[1]):
            pct = t / total * 100 if total > 0 else 0
            print(f"  {name:12}: {t:8.2f}s ({pct:5.1f}%)")

    def get_gf(self, n: int) -> Tuple[object, GaloisField]:
        """Get or create Galois field instances for given n."""
        if n not in self._galois_gf:
            self._galois_gf[n] = galois.GF(2 ** n)
            self._pm_gf[n] = GaloisField(n)
        return self._galois_gf[n], self._pm_gf[n]

    def get_mcp_solver(self, n: int) -> MCPSolver:
        """Get or create MCP solver for given n."""
        if n not in self._mcp_solvers:
            _, pm_gf = self.get_gf(n)
            self._mcp_solvers[n] = MCPSolver(gf=pm_gf, n=n, verbose=False)
        return self._mcp_solvers[n]

    def get_decoder(self, model_name: str, secret_key: str, n: int, gf: object,
                    hamming_mode: str, correct: bool, green_fraction: float,
                    device: str) -> LLMWatermarkDecoder:
        """Get or create a decoder for the given parameters.

        IMPORTANT: device must match the device used during encoding.
        torch.randperm produces different results on CPU vs CUDA.
        """
        # Cache key based on model (tokenizer is model-specific)
        cache_key = model_name

        if cache_key not in self._decoders:
            self._decoders[cache_key] = LLMWatermarkDecoder(
                model_name=model_name,
                secret_key=secret_key,
                n=n,
                gf=gf,
                hamming_mode=hamming_mode,
                correct=correct,
                green_list_fraction=green_fraction,
                device=device,
                verbose=False
            )

        # Update secret_key and other params for this specific row
        decoder = self._decoders[cache_key]
        decoder.secret_key = secret_key
        decoder.n = n
        decoder.gf = gf
        decoder.hamming_mode = hamming_mode
        decoder.correct = correct
        decoder.green_list_fraction = green_fraction
        decoder.device = device

        # Update hamming instance if mode changed
        if hamming_mode != "none":
            from src.hamming import HammingCode
            decoder.hamming = HammingCode(n, secded=(hamming_mode == "secded"))
        else:
            decoder.hamming = None

        return decoder


# =============================================================================
# Multiprocessing Support
# =============================================================================

# Worker process state (initialized once per worker)
_worker_state: Dict[str, Any] = {}


class RowWrapper:
    """Wrapper to provide dict-like access to row data for multiprocessing."""

    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)


def _init_worker(device: str) -> None:
    """Initialize worker process with its own decoder cache.

    Called once per worker process at pool creation.
    """
    global _worker_state
    _worker_state = {
        'cache': SimulationCache(),
        'device': device,
    }


def _process_row_task(task: Dict) -> Tuple[List[Dict], Dict[str, float]]:
    """Process a single row in a worker process.

    Args:
        task: Dict containing row data and configuration

    Returns:
        Tuple of (result dicts for this row, timing dict)
    """
    global _worker_state

    cache = _worker_state['cache']
    device = _worker_state['device']

    # Reset timing for this row
    cache.timing = {'gf_mcp': 0.0, 'decoder': 0.0, 'attack': 0.0, 'decode': 0.0, 'mcp': 0.0}

    # Wrap row data for dict-like access
    row = RowWrapper(task['row'])
    rng = random.Random(task['seed'])

    # Run simulation for this row
    results = run_simulation_for_row(
        row=row,
        config=task['config'],
        cache=cache,
        budget=task['budget'],
        groups_list=task['groups_list'],
        device=device,
        rng=rng
    )

    # Add metadata to results
    for r in results:
        r['model'] = task['model']
        r['row_idx'] = task['row_idx']
        r['tokens_length'] = task['tokens_length']
        r['budget'] = task['budget']
        r['budget_pct'] = task['budget'] / task['tokens_length'] * 100 if task['tokens_length'] > 0 else 0

    return results, cache.timing.copy()


# =============================================================================
# Core Simulation
# =============================================================================

def run_simulation_for_row(
    row: pd.Series,
    config: Dict,
    cache: SimulationCache,
    budget: int,
    groups_list: List[int],
    device: str,
    rng: random.Random
) -> List[Dict]:
    """
    Run attack simulation for a single data row.

    Args:
        row: DataFrame row with generated_ids, a0, a1, etc.
        config: Experiment config dict
        cache: Simulation cache instance
        budget: Fixed number of tokens to attack
        groups_list: List of group counts to test
        device: Device for decoder ('cuda' or 'cpu')
        rng: Random number generator

    Returns:
        List of result dicts, one per (groups, attack_type) combination
    """
    results = []

    # Parse row data
    generated_ids = json.loads(row['generated_ids'])
    a0 = int(row['a0'])
    a1 = int(row['a1'])
    secret_key = row['secret_key']

    # Get config params
    n = config.get('n', 8)
    hamming_mode = config.get('hamming', 'none')
    correct = config.get('correct', False)
    green_fraction = config.get('green_fraction', 0.5)
    model_name = config.get('model', 'unknown')

    # Get cached GF instances and MCP solver
    t0 = time.perf_counter()
    gf, _ = cache.get_gf(n)
    mcp_solver = cache.get_mcp_solver(n)
    cache.timing['gf_mcp'] += time.perf_counter() - t0

    # Get decoder
    t0 = time.perf_counter()
    decoder = cache.get_decoder(
        model_name=model_name,
        secret_key=secret_key,
        n=n,
        gf=gf,
        hamming_mode=hamming_mode,
        correct=correct if hamming_mode != "none" else False,
        green_fraction=green_fraction,
        device=device
    )
    cache.timing['decoder'] += time.perf_counter() - t0

    vocab_size = decoder.vocab_size

    # Parse watermark blocks for matching calculation
    watermark_blocks = json.loads(row['watermark_blocks']) if row.get('watermark_blocks') else []

    # Baseline (groups=0): Use original stats from data (no attack)
    baseline_recovered = bool(row['watermark_recovered'])
    baseline_valid = int(row.get('unique_valid_blocks_count', 0))
    baseline_matching = int(row.get('unique_matching_blocks_count', 0))

    attack_types = ['insertion', 'deletion', 'substitution']

    # Add baseline for all attack types (groups=0 means no attack)
    for attack_type in attack_types:
        results.append({
            'groups': 0,
            'attack_type': attack_type,
            'recovered': baseline_recovered,
            'valid_blocks': baseline_valid,
            'matching_blocks': baseline_matching
        })

    # For groups > 0: run attacks with fixed budget split into N groups
    for num_groups in groups_list:
        # Can't have more groups than budget
        effective_groups = min(num_groups, budget)
        group_sizes = split_budget(budget, effective_groups)

        for attack_type in attack_types:
            t0 = time.perf_counter()

            # For deletion, ensure we don't delete more than len-1 tokens
            if attack_type == 'deletion':
                max_deletable = len(generated_ids) - 1
                if budget > max_deletable:
                    # Reduce budget for this specific attack
                    adjusted_budget = max_deletable
                    adjusted_groups = min(effective_groups, max_deletable)
                    adjusted_sizes = split_budget(adjusted_budget, adjusted_groups)
                else:
                    adjusted_sizes = group_sizes
                attack_group_sizes = adjusted_sizes
            else:
                attack_group_sizes = group_sizes

            # Skip if no valid attack possible
            if not attack_group_sizes or sum(attack_group_sizes) == 0:
                results.append({
                    'groups': num_groups,
                    'attack_type': attack_type,
                    'recovered': False,
                    'valid_blocks': 0,
                    'matching_blocks': 0
                })
                continue

            # Compute positions based on attack type
            try:
                if attack_type == 'insertion':
                    # For insertion, positions are in original text (can insert at len+1 positions)
                    positions = compute_group_positions(len(generated_ids) + 1, attack_group_sizes, rng)
                else:
                    # For deletion/substitution, positions are token indices
                    positions = compute_group_positions(len(generated_ids), attack_group_sizes, rng)
            except ValueError:
                # Groups don't fit
                results.append({
                    'groups': num_groups,
                    'attack_type': attack_type,
                    'recovered': False,
                    'valid_blocks': 0,
                    'matching_blocks': 0
                })
                continue

            # Apply attack
            if attack_type == 'insertion':
                attacked_ids = insertion_attack(generated_ids, attack_group_sizes, positions, vocab_size, rng)
            elif attack_type == 'deletion':
                attacked_ids = deletion_attack(generated_ids, attack_group_sizes, positions, rng)
            else:  # substitution
                attacked_ids = substitution_attack(generated_ids, attack_group_sizes, positions, vocab_size, rng)
            cache.timing['attack'] += time.perf_counter() - t0

            # Skip if attacked list is too short
            if len(attacked_ids) < decoder.tokens_per_block:
                results.append({
                    'groups': num_groups,
                    'attack_type': attack_type,
                    'recovered': False,
                    'valid_blocks': 0,
                    'matching_blocks': 0
                })
                continue

            # Decode attacked tokens
            t0 = time.perf_counter()
            all_blocks, valid_blocks, _ = decoder.decode_text(generated_ids=attacked_ids)
            cache.timing['decode'] += time.perf_counter() - t0

            # Use valid_blocks for Hamming mode, all_blocks otherwise
            blocks_for_mcp = valid_blocks if hamming_mode != "none" else all_blocks

            # Verify watermark
            t0 = time.perf_counter()
            verification = mcp_solver.verify_watermark(
                decoded_blocks=blocks_for_mcp,
                original_a0=a0,
                original_a1=a1,
                watermark_blocks=watermark_blocks
            )
            elapsed = time.perf_counter() - t0
            cache.timing['mcp'] += elapsed

            # Count unique valid/matching blocks
            unique_valid = len(set((b['x'], b['y']) for b in blocks_for_mcp))
            unique_matching = len(set((b['x'], b['y']) for b in verification.get('matching_blocks', [])))

            results.append({
                'groups': num_groups,
                'attack_type': attack_type,
                'recovered': verification['is_valid'],
                'valid_blocks': unique_valid,
                'matching_blocks': unique_matching
            })

    return results


def run_simulation(
    prepared_data: Dict,
    perturbation_pct: int,
    groups_list: List[int],
    device: str,
    seed: int
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Run attack simulation across all models and rows.

    Args:
        prepared_data: Output from load_and_prepare_experiments()
        perturbation_pct: Perturbation rate as percentage (fixed budget)
        groups_list: List of group counts to test
        device: Device for decoder ('cuda' or 'cpu')
        seed: Random seed

    Returns:
        Tuple of (DataFrame with simulation results, timing dict)
    """
    cache = SimulationCache()
    all_results = []
    total_timing = {'gf_mcp': 0.0, 'decoder': 0.0, 'attack': 0.0, 'decode': 0.0, 'mcp': 0.0}

    for model, model_data in prepared_data.items():
        cache.timing = {'gf_mcp': 0.0, 'decoder': 0.0, 'attack': 0.0, 'decode': 0.0, 'mcp': 0.0}  # Reset per model
        df = model_data['df']
        config = model_data['config']

        print(f"\nSimulating attacks for model: {model}")
        print(f"  Rows: {len(df)}")

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  Processing"):
            # Calculate fixed budget for this row
            tokens_length = int(row['tokens_length'])
            budget = max(1, int(tokens_length * perturbation_pct / 100))

            # Deterministic RNG per row (matches parallel mode)
            row_rng = random.Random(seed + idx)

            # Run simulation
            row_results = run_simulation_for_row(
                row=row,
                config=config,
                cache=cache,
                budget=budget,
                groups_list=groups_list,
                device=device,
                rng=row_rng
            )

            # Add row metadata to results
            for result in row_results:
                result['model'] = model
                result['row_idx'] = idx
                result['tokens_length'] = tokens_length
                result['budget'] = budget
                result['budget_pct'] = budget / tokens_length * 100 if tokens_length > 0 else 0
                all_results.append(result)

        cache.print_timing()

        # Accumulate timing across models
        for key in total_timing:
            total_timing[key] += cache.timing[key]

    return pd.DataFrame(all_results), total_timing


def run_simulation_parallel(
    prepared_data: Dict,
    perturbation_pct: int,
    groups_list: List[int],
    device: str,
    seed: int,
    num_workers: int
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Run attack simulation in parallel across all models and rows.

    Args:
        prepared_data: Output from load_and_prepare_experiments()
        perturbation_pct: Perturbation rate as percentage (fixed budget)
        groups_list: List of group counts to test
        device: Device for decoder ('cuda' or 'cpu')
        seed: Random seed
        num_workers: Number of worker processes

    Returns:
        Tuple of (DataFrame with simulation results, aggregated timing dict)
    """
    all_results = []
    total_timing = {'gf_mcp': 0.0, 'decoder': 0.0, 'attack': 0.0, 'decode': 0.0, 'mcp': 0.0}

    for model, model_data in prepared_data.items():
        df = model_data['df']
        config = model_data['config']

        print(f"\nSimulating attacks for model: {model}")
        print(f"  Rows: {len(df)}, Workers: {num_workers}")

        # Prepare tasks - one per row
        tasks = []
        for idx, row in df.iterrows():
            tokens_length = int(row['tokens_length'])
            budget = max(1, int(tokens_length * perturbation_pct / 100))
            row_seed = seed + idx  # Deterministic seed per row

            tasks.append({
                'row': row.to_dict(),
                'config': config,
                'budget': budget,
                'groups_list': groups_list,
                'model': model,
                'row_idx': idx,
                'tokens_length': tokens_length,
                'seed': row_seed,
            })

        # Process in parallel
        # Use 'spawn' on Windows/macOS for CUDA compatibility
        ctx = mp.get_context('spawn')
        with ctx.Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(device,)
        ) as pool:
            for results, row_timing in tqdm(
                pool.imap_unordered(_process_row_task, tasks),
                total=len(tasks),
                desc=f"  Processing"
            ):
                all_results.extend(results)
                # Aggregate timing
                for key in total_timing:
                    total_timing[key] += row_timing.get(key, 0.0)

    return pd.DataFrame(all_results), total_timing


# =============================================================================
# Reporting
# =============================================================================

def format_summary_report(results_df: pd.DataFrame, args, groups_list: List[int],
                          timing: dict = None, source_dirs: List[str] = None,
                          elapsed: float = None) -> str:
    """Generate text summary of attack simulation results."""
    lines = []
    w = 80

    lines.append('=' * w)
    lines.append('ATTACK SIMULATION REPORT')
    lines.append('=' * w)
    lines.append(f'Perturbation Rate: {args.perturbation_rate}%')
    lines.append(f'Groups Tested: {groups_list}')
    lines.append(f'Seed: {args.seed}')
    lines.append(f'Total Simulations: {len(results_df)}')

    # List source directories if available
    if source_dirs and len(source_dirs) > 0:
        lines.append('')
        lines.append('-' * w)
        lines.append(f'SOURCE EXPERIMENTS ({len(source_dirs)})')
        lines.append('-' * w)
        for src in sorted(source_dirs):
            lines.append(f'  {src}')

    # Add timing breakdown if available
    if timing:
        lines.append('')
        lines.append('-' * w)
        lines.append('TIMING BREAKDOWN (aggregate CPU time)')
        lines.append('-' * w)
        total = sum(timing.values())
        for name, t in sorted(timing.items(), key=lambda x: -x[1]):
            pct = t / total * 100 if total > 0 else 0
            lines.append(f"  {name:12}: {t:8.2f}s ({pct:5.1f}%)")
        lines.append(f"  {'TOTAL':12}: {total:8.2f}s")
    if elapsed is not None:
        lines.append('')
        lines.append(f"Elapsed time: {elapsed:.1f}s")
    lines.append('')

    # Summary per attack type
    attack_types = ['insertion', 'deletion', 'substitution']

    # Get all unique group counts, sorted
    all_groups = sorted(results_df['groups'].unique())

    lines.append('-' * w)
    lines.append('RECOVERY RATE BY ATTACK TYPE AND GROUP COUNT')
    lines.append('-' * w)

    # Build header with all group counts
    header = f"{'Attack Type':<15}"
    for g in all_groups:
        if g == 0:
            header += f"{'Baseline':>10}"
        else:
            header += f"{'G=' + str(g):>10}"
    header += f"{'Drop':>10}"
    lines.append(header)

    for attack_type in attack_types:
        attack_data = results_df[results_df['attack_type'] == attack_type]

        row = f"  {attack_type.capitalize():<13}"
        rates = []
        for g in all_groups:
            rate = attack_data[attack_data['groups'] == g]['recovered'].mean() * 100
            rates.append(rate)
            row += f"{rate:>9.1f}%"

        # Calculate drop from baseline to max groups
        if len(rates) >= 2:
            drop = rates[0] - rates[-1]
            row += f"{drop:>9.1f}%"
        else:
            row += f"{'N/A':>10}"

        lines.append(row)

    lines.append('')
    lines.append('=' * w)

    return '\n'.join(lines)


def generate_recovery_graph(results_df: pd.DataFrame, output_path: Path,
                            perturbation_pct: int) -> None:
    """Generate PNG with 3 subplots showing recovery rate vs groups for each attack type.

    Args:
        results_df: DataFrame with simulation results
        output_path: Path to save the PNG file
        perturbation_pct: Perturbation rate used (for title)
    """
    attack_types = ['insertion', 'deletion', 'substitution']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Watermark Recovery Rate vs Attack Groups (Budget: {perturbation_pct}% of tokens)',
                 fontsize=14, fontweight='bold')

    for ax, attack_type in zip(axes, attack_types):
        attack_data = results_df[results_df['attack_type'] == attack_type]

        # Calculate recovery rate per group count
        recovery_by_groups = attack_data.groupby('groups')['recovered'].mean() * 100
        groups = recovery_by_groups.index.tolist()
        rates = recovery_by_groups.values.tolist()

        # Plot
        ax.plot(groups, rates, marker='o', linewidth=2, markersize=6)
        ax.set_xlabel('Number of Attack Groups', fontsize=11)
        ax.set_ylabel('Recovery Rate (%)', fontsize=11)
        ax.set_title(attack_type.capitalize(), fontsize=12, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)

        # Add baseline annotation
        if 0 in groups:
            baseline_rate = recovery_by_groups[0]
            ax.axhline(y=baseline_rate, color='green', linestyle='--', alpha=0.5)
            # Label inside plot, right-aligned
            ax.text(max(groups), baseline_rate, f'{baseline_rate:.1f}% ',
                    va='bottom', ha='right', color='green', fontsize=9)

        # Set x-axis ticks to integers only
        ax.set_xticks(groups)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved graph: {output_path}")


def generate_combined_recovery_graph(results_df: pd.DataFrame, output_path: Path,
                                     perturbation_pct: int) -> None:
    """Generate PNG with all attack types on a single plot.

    Args:
        results_df: DataFrame with simulation results
        output_path: Path to save the PNG file
        perturbation_pct: Perturbation rate used (for title)
    """
    attack_types = ['insertion', 'deletion', 'substitution']
    colors = {'insertion': 'tab:blue', 'deletion': 'tab:orange', 'substitution': 'tab:red'}

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(f'Watermark Recovery Rate vs Attack Groups (Budget: {perturbation_pct}% of tokens)',
                 fontsize=14, fontweight='bold')

    baseline_rate = None

    for attack_type in attack_types:
        attack_data = results_df[results_df['attack_type'] == attack_type]

        # Calculate recovery rate per group count
        recovery_by_groups = attack_data.groupby('groups')['recovered'].mean() * 100
        groups = recovery_by_groups.index.tolist()
        rates = recovery_by_groups.values.tolist()

        # Store baseline (same for all attack types)
        if baseline_rate is None and 0 in groups:
            baseline_rate = recovery_by_groups[0]

        # Plot
        ax.plot(groups, rates, marker='o', linewidth=2, markersize=6,
                color=colors[attack_type], label=attack_type.capitalize())

    ax.set_xlabel('Number of Attack Groups', fontsize=11)
    ax.set_ylabel('Recovery Rate (%)', fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    # Add baseline
    if baseline_rate is not None:
        ax.axhline(y=baseline_rate, color='green', linestyle='--', alpha=0.5)
        ax.text(max(groups), baseline_rate, f'{baseline_rate:.1f}% ',
                va='bottom', ha='right', color='green', fontsize=9)

    # Set x-axis ticks to integers only
    ax.set_xticks(groups)

    # Legend in bottom left
    ax.legend(loc='lower left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved graph: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    print("Attack Simulation Script")
    print("=" * 40)

    # Determine device
    device = get_device(args.no_cuda)
    print(f"Device: {device}")

    # Parse groups argument
    try:
        groups_list = parse_groups_arg(args.groups)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    print(f"Groups to test: {groups_list}")

    # Validate perturbation rate
    if args.perturbation_rate >= 100:
        print("Error: --perturbation-rate must be < 100")
        return 1

    # Determine number of workers
    if args.workers == 0:
        num_workers = min(cpu_count(), 8)  # Auto: use up to 8 cores
    else:
        num_workers = args.workers

    print(f"Workers: {num_workers}" + (" (serial)" if num_workers == 1 else " (parallel)"))

    # Set random seed
    random.seed(args.seed)

    # Load and prepare experiments
    prepared_data = load_and_prepare_experiments(
        min_tokens=args.min_tokens,
        force=args.force,
        verbose=True
    )

    # Run simulation (serial or parallel)
    t_start = time.perf_counter()
    if num_workers == 1:
        results_df, timing = run_simulation(
            prepared_data=prepared_data,
            perturbation_pct=args.perturbation_rate,
            groups_list=groups_list,
            device=device,
            seed=args.seed
        )
    else:
        results_df, timing = run_simulation_parallel(
            prepared_data=prepared_data,
            perturbation_pct=args.perturbation_rate,
            groups_list=groups_list,
            device=device,
            seed=args.seed,
            num_workers=num_workers
        )
    elapsed = time.perf_counter() - t_start
    print(f"\nSimulation completed in {elapsed:.1f}s")

    # Collect all source directories
    all_source_dirs = []
    for model_data in prepared_data.values():
        all_source_dirs.extend(model_data.get('sources', []))

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy source configs
    if all_source_dirs:
        configs_dir = output_dir / 'source_configs'
        configs_dir.mkdir(exist_ok=True)
        for source_dir in all_source_dirs:
            config_src = Path('output') / source_dir / 'run_config.json'
            if config_src.exists():
                config_dst = configs_dir / f'{source_dir}.json'
                shutil.copy(config_src, config_dst)

    # Save results
    results_path = output_dir / "attack_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved results: {results_path}")

    # Generate recovery rate graphs
    graph_path = output_dir / "recovery_rate.png"
    generate_recovery_graph(results_df, graph_path, args.perturbation_rate)

    combined_graph_path = output_dir / "recovery_rate_combined.png"
    generate_combined_recovery_graph(results_df, combined_graph_path, args.perturbation_rate)

    # Generate and save summary
    summary = format_summary_report(results_df, args, groups_list, timing, all_source_dirs, elapsed)
    summary_path = output_dir / "attack_summary.txt"
    summary_path.write_text(summary)
    print(f"Saved summary: {summary_path}")

    # Print summary to console
    print()
    print(summary)

    print(f"\nAttack simulation complete. Results saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
