"""
Attack simulation script for watermark robustness testing.

Simulates insertion, deletion, and substitution attacks on watermarked text
and measures watermark recovery rate at various attack budgets.
"""
import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import galois

# Add parent directory to path for src imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib import load_and_prepare_experiments
from src.llm_watermark import LLMWatermarkDecoder, MCPSolver
from src.pm_galois import GaloisField


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
    return parser.parse_args()


# =============================================================================
# Attack Functions
# =============================================================================

def insertion_attack(token_ids: List[int], num_insertions: int, vocab_size: int,
                     rng: random.Random) -> List[int]:
    """
    Insert random tokens at random positions.

    Args:
        token_ids: Original token IDs
        num_insertions: Number of tokens to insert
        vocab_size: Vocabulary size for random token generation
        rng: Random number generator

    Returns:
        Modified token IDs with insertions
    """
    result = token_ids.copy()
    for _ in range(num_insertions):
        pos = rng.randint(0, len(result))
        token = rng.randint(0, vocab_size - 1)
        result.insert(pos, token)
    return result


def deletion_attack(token_ids: List[int], num_deletions: int,
                    rng: random.Random) -> List[int]:
    """
    Delete tokens at random positions.

    Args:
        token_ids: Original token IDs
        num_deletions: Number of tokens to delete
        rng: Random number generator

    Returns:
        Modified token IDs with deletions
    """
    result = token_ids.copy()
    # Cap deletions to avoid empty list
    actual_deletions = min(num_deletions, len(result) - 1)
    for _ in range(actual_deletions):
        if len(result) > 1:
            pos = rng.randint(0, len(result) - 1)
            del result[pos]
    return result


def substitution_attack(token_ids: List[int], num_substitutions: int, vocab_size: int,
                        rng: random.Random) -> List[int]:
    """
    Replace tokens at random positions with random tokens.

    Args:
        token_ids: Original token IDs
        num_substitutions: Number of tokens to substitute
        vocab_size: Vocabulary size for random token generation
        rng: Random number generator

    Returns:
        Modified token IDs with substitutions
    """
    result = token_ids.copy()
    # Cap substitutions to list length
    actual_subs = min(num_substitutions, len(result))
    positions = rng.sample(range(len(result)), actual_subs)
    for pos in positions:
        result[pos] = rng.randint(0, vocab_size - 1)
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
                    hamming_mode: str, correct: bool, green_fraction: float) -> LLMWatermarkDecoder:
        """Get or create a decoder for the given parameters."""
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

        # Update hamming instance if mode changed
        if hamming_mode != "none":
            from src.hamming import HammingCode
            decoder.hamming = HammingCode(n, secded=(hamming_mode == "secded"))
        else:
            decoder.hamming = None

        return decoder


# =============================================================================
# Core Simulation
# =============================================================================

def run_simulation_for_row(
    row: pd.Series,
    config: Dict,
    cache: SimulationCache,
    max_budget: int,
    rng: random.Random
) -> List[Dict]:
    """
    Run attack simulation for a single data row.

    Args:
        row: DataFrame row with generated_ids, a0, a1, etc.
        config: Experiment config dict
        cache: Simulation cache instance
        max_budget: Maximum number of bits to attack
        rng: Random number generator

    Returns:
        List of result dicts, one per (budget, attack_type) combination
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
        green_fraction=green_fraction
    )
    cache.timing['decoder'] += time.perf_counter() - t0

    vocab_size = decoder.vocab_size

    # Parse watermark blocks for matching calculation
    watermark_blocks = json.loads(row['watermark_blocks']) if row.get('watermark_blocks') else []

    # Budget 0: Use original stats from data (no simulation needed)
    baseline_result = {
        'budget': 0,
        'attack_type': 'baseline',
        'recovered': bool(row['watermark_recovered']),
        'valid_blocks': int(row.get('unique_valid_blocks', 0)),
        'matching_blocks': int(row.get('unique_matching_blocks', 0))
    }

    # For budget > 0: run attacks
    attack_types = ['insertion', 'deletion', 'substitution']

    for budget in range(1, max_budget + 1):
        for attack_type in attack_types:
            # Apply attack
            t0 = time.perf_counter()
            if attack_type == 'insertion':
                attacked_ids = insertion_attack(generated_ids, budget, vocab_size, rng)
            elif attack_type == 'deletion':
                attacked_ids = deletion_attack(generated_ids, budget, rng)
            else:  # substitution
                attacked_ids = substitution_attack(generated_ids, budget, vocab_size, rng)
            cache.timing['attack'] += time.perf_counter() - t0

            # Skip if attacked list is too short
            if len(attacked_ids) < decoder.tokens_per_block:
                results.append({
                    'budget': budget,
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
                'budget': budget,
                'attack_type': attack_type,
                'recovered': verification['is_valid'],
                'valid_blocks': unique_valid,
                'matching_blocks': unique_matching
            })

    # Add baseline for all attack types (they all start from same point)
    for attack_type in attack_types:
        results.append({
            'budget': 0,
            'attack_type': attack_type,
            'recovered': baseline_result['recovered'],
            'valid_blocks': baseline_result['valid_blocks'],
            'matching_blocks': baseline_result['matching_blocks']
        })

    return results


def run_simulation(
    prepared_data: Dict,
    max_perturbation_pct: int,
    seed: int
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Run attack simulation across all models and rows.

    Args:
        prepared_data: Output from load_and_prepare_experiments()
        max_perturbation_pct: Maximum perturbation rate as percentage
        seed: Random seed

    Returns:
        Tuple of (DataFrame with simulation results, timing dict)
    """
    rng = random.Random(seed)
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
            # Calculate max budget for this row
            tokens_length = int(row['tokens_length'])
            max_budget = max(1, int(tokens_length * max_perturbation_pct / 100))

            # Run simulation
            row_results = run_simulation_for_row(
                row=row,
                config=config,
                cache=cache,
                max_budget=max_budget,
                rng=rng
            )

            # Add row metadata to results
            for result in row_results:
                result['model'] = model
                result['row_idx'] = idx
                result['tokens_length'] = tokens_length
                result['max_budget'] = max_budget
                result['budget_pct'] = result['budget'] / tokens_length * 100 if tokens_length > 0 else 0
                all_results.append(result)

        cache.print_timing()

        # Accumulate timing across models
        for key in total_timing:
            total_timing[key] += cache.timing[key]

    return pd.DataFrame(all_results), total_timing


# =============================================================================
# Visualization
# =============================================================================

def create_match_rate_plot(results_df: pd.DataFrame, output_path: Path):
    """
    Create line plot of match rate vs attack budget.

    X-axis: Budget (number of attacked bits)
    Y-axis: Match rate (0-100%)
    3 lines: Insertion, Deletion, Substitution
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    attack_types = ['insertion', 'deletion', 'substitution']
    colors = {'insertion': 'blue', 'deletion': 'red', 'substitution': 'green'}

    for attack_type in attack_types:
        attack_data = results_df[results_df['attack_type'] == attack_type]

        # Group by budget and calculate mean match rate
        grouped = attack_data.groupby('budget')['recovered'].mean() * 100

        ax.plot(grouped.index, grouped.values,
                label=attack_type.capitalize(),
                color=colors[attack_type],
                marker='o', markersize=3, linewidth=1.5)

    ax.set_xlabel('Attack Budget (bits)', fontsize=12)
    ax.set_ylabel('Watermark Recovery Rate (%)', fontsize=12)
    ax.set_title('Watermark Recovery Rate Under Attack', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def create_blocks_plot(results_df: pd.DataFrame, output_path: Path):
    """
    Create line plot of block counts vs attack budget.

    Shows valid_blocks and matching_blocks for each attack type.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    attack_types = ['insertion', 'deletion', 'substitution']

    for ax, attack_type in zip(axes, attack_types):
        attack_data = results_df[results_df['attack_type'] == attack_type]

        # Group by budget
        grouped = attack_data.groupby('budget').agg({
            'valid_blocks': 'mean',
            'matching_blocks': 'mean'
        })

        ax.plot(grouped.index, grouped['valid_blocks'],
                label='Valid Blocks', color='blue', marker='o', markersize=3)
        ax.plot(grouped.index, grouped['matching_blocks'],
                label='Matching Blocks', color='orange', marker='s', markersize=3)

        ax.set_xlabel('Attack Budget (bits)')
        ax.set_ylabel('Average Block Count')
        ax.set_title(f'{attack_type.capitalize()} Attack')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle('Block Counts Under Attack', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def format_summary_report(results_df: pd.DataFrame, args, timing: dict = None) -> str:
    """Generate text summary of attack simulation results."""
    lines = []
    w = 80

    lines.append('=' * w)
    lines.append('ATTACK SIMULATION REPORT')
    lines.append('=' * w)
    lines.append(f'Perturbation Rate: {args.perturbation_rate}%')
    lines.append(f'Seed: {args.seed}')
    lines.append(f'Total Simulations: {len(results_df)}')

    # Add timing breakdown if available
    if timing:
        lines.append('')
        lines.append('-' * w)
        lines.append('TIMING BREAKDOWN')
        lines.append('-' * w)
        total = sum(timing.values())
        for name, t in sorted(timing.items(), key=lambda x: -x[1]):
            pct = t / total * 100 if total > 0 else 0
            lines.append(f"  {name:12}: {t:8.2f}s ({pct:5.1f}%)")
        lines.append(f"  {'TOTAL':12}: {total:8.2f}s")
    lines.append('')

    # Summary per attack type
    attack_types = ['insertion', 'deletion', 'substitution']

    lines.append('-' * w)
    lines.append('RECOVERY RATE BY ATTACK TYPE')
    lines.append('-' * w)
    lines.append(f"{'Attack Type':<15} {'Budget=0':<12} {'Budget=Max':<12} {'Avg Drop':<12}")

    for attack_type in attack_types:
        attack_data = results_df[results_df['attack_type'] == attack_type]

        budget_0 = attack_data[attack_data['budget'] == 0]['recovered'].mean() * 100
        max_budget = attack_data['budget'].max()
        budget_max = attack_data[attack_data['budget'] == max_budget]['recovered'].mean() * 100
        drop = budget_0 - budget_max

        lines.append(f"  {attack_type.capitalize():<13} {budget_0:>10.1f}% {budget_max:>10.1f}% {drop:>10.1f}%")

    lines.append('')
    lines.append('=' * w)

    return '\n'.join(lines)


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    print("Attack Simulation Script")
    print("=" * 40)

    # Set random seed
    random.seed(args.seed)

    # Load and prepare experiments
    prepared_data = load_and_prepare_experiments(
        min_tokens=args.min_tokens,
        force=args.force,
        verbose=True
    )

    # Run simulation
    results_df, timing = run_simulation(
        prepared_data=prepared_data,
        max_perturbation_pct=args.perturbation_rate,
        seed=args.seed
    )

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    results_path = output_dir / "attack_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved results: {results_path}")

    # Generate plots
    match_rate_path = output_dir / "match_rate_vs_budget.png"
    create_match_rate_plot(results_df, match_rate_path)
    print(f"Saved plot: {match_rate_path}")

    blocks_path = output_dir / "blocks_vs_budget.png"
    create_blocks_plot(results_df, blocks_path)
    print(f"Saved plot: {blocks_path}")

    # Generate and save summary
    summary = format_summary_report(results_df, args, timing)
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
