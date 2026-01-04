"""
TP Count Histogram - Visualize true positive counts by collinear points level.

For verified watermarked samples (where recovered a0/a1 match original):
- Count samples with exactly N collinear points (N = 2, 3, 4, ...)
- Generates bar chart with collinear points on X-axis, true positives on Y-axis
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from multiprocessing import cpu_count
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from lib import load_and_prepare_experiments
from src.pm_galois import GaloisField, max_collinear_points, recover_line_equation


# =============================================================================
# Multiprocessing Support
# =============================================================================

# Worker process state (initialized once per worker)
_worker_gf: Optional[GaloisField] = None


def _init_worker(n: int) -> None:
    """Initialize worker process with its own GaloisField instance."""
    global _worker_gf
    _worker_gf = GaloisField(n)


def _process_sample(sample: Dict) -> Optional[int]:
    """
    Process a single sample in a worker process.

    Returns:
        Collinear count if verified, None otherwise
    """
    global _worker_gf

    blocks = sample['blocks']
    a0_orig = sample['a0']
    a1_orig = sample['a1']

    # Run MCP
    count, slope, collinear_pts = max_collinear_points(blocks, _worker_gf)

    # Verify line recovery
    if count >= 2 and len(collinear_pts) >= 2:
        try:
            recovered_a0, recovered_a1 = recover_line_equation(collinear_pts, _worker_gf)
            if recovered_a0 == a0_orig and recovered_a1 == a1_orig:
                return count
        except ValueError:
            pass

    return None


class GFCache:
    """Cache GaloisField instances to avoid recomputing inverse tables."""

    def __init__(self):
        self._cache: Dict[int, GaloisField] = {}

    def get(self, n: int) -> GaloisField:
        if n not in self._cache:
            self._cache[n] = GaloisField(n)
        return self._cache[n]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate true positive count histogram for watermark detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tp_count_histogram.py                          # Auto-discover from output/
  python tp_count_histogram.py --input-dir output/exp1  # Specific experiment
  python tp_count_histogram.py --min-tokens 200         # Filter by min tokens
  python tp_count_histogram.py --verbose                # Show progress bars
        """
    )
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Input directory (experiment folder or parent)")
    parser.add_argument("--min-tokens", type=int, default=None,
                        help="Min tokens filter")
    parser.add_argument("--force", action="store_true",
                        help="Proceed despite parameter conflicts")
    parser.add_argument("--verbose", action="store_true",
                        help="Show progress bars")
    parser.add_argument("--output-dir", type=str, default="output/tp_count_histogram",
                        help="Output directory (default: output/tp_count_histogram)")
    parser.add_argument("-j", "--workers", type=int, default=1,
                        help="Number of worker processes (default: 1, 0=auto)")
    return parser.parse_args()


def extract_samples(df: pd.DataFrame, config: Dict, verbose: bool = False) -> List[Dict]:
    """
    Parse valid_blocks JSON, extract (x, y) points and original coefficients.

    Args:
        df: DataFrame with valid_blocks, a0, a1 columns
        config: Experiment config (contains 'n')
        verbose: Print skip statistics

    Returns:
        List of sample dicts with 'blocks', 'a0', 'a1', 'n', 'row_idx'
    """
    n = config.get('n', 8)
    field_size = 2 ** n

    samples = []
    skipped_empty = 0
    skipped_small = 0

    for idx, row in df.iterrows():
        # Parse valid_blocks JSON
        try:
            blocks_json = row['valid_blocks']
            if pd.isna(blocks_json) or blocks_json == '':
                skipped_empty += 1
                continue
            blocks_list = json.loads(blocks_json)
        except (json.JSONDecodeError, TypeError):
            skipped_empty += 1
            continue

        # Extract (x, y) tuples
        points = [(b['x'], b['y']) for b in blocks_list]

        if len(points) < 2:
            skipped_small += 1
            continue

        samples.append({
            'blocks': points,
            'block_count': len(points),
            'n': n,
            'a0': int(row['a0']),
            'a1': int(row['a1']),
            'row_idx': idx
        })

    if verbose:
        print(f"    Extracted: {len(samples)} samples")
        print(f"    Skipped (empty): {skipped_empty}")
        print(f"    Skipped (< 2 blocks): {skipped_small}")

    return samples


def compute_mcp_verified(
    samples: List[Dict],
    gf_cache: GFCache,
    verbose: bool = False,
    num_workers: int = 1
) -> np.ndarray:
    """
    Run MCP on samples and verify line recovery.

    Args:
        samples: List of sample dicts
        gf_cache: GaloisField cache
        verbose: Show progress bar
        num_workers: Number of worker processes (1=sequential)

    Returns:
        Array of max_collinear_count for VERIFIED samples only
        (where recovered_a0 == a0 and recovered_a1 == a1)
    """
    if not samples:
        return np.array([])

    n = samples[0]['n']  # All samples have the same n

    # Parallel processing
    if num_workers != 1:
        if num_workers == 0:
            num_workers = min(cpu_count(), 8)

        ctx = mp.get_context('spawn')
        with ctx.Pool(num_workers, initializer=_init_worker, initargs=(n,)) as pool:
            results = list(tqdm(
                pool.imap(_process_sample, samples),
                total=len(samples),
                desc="MCP",
                disable=not verbose
            ))

        verified_scores = [r for r in results if r is not None]
        return np.array(verified_scores)

    # Sequential processing
    verified_scores = []
    gf = gf_cache.get(n)

    iterator = tqdm(samples, desc="MCP", disable=not verbose)
    for sample in iterator:
        blocks = sample['blocks']
        a0_orig = sample['a0']
        a1_orig = sample['a1']

        # Run MCP
        count, slope, collinear_pts = max_collinear_points(blocks, gf)

        # Verify line recovery
        if count >= 2 and len(collinear_pts) >= 2:
            try:
                recovered_a0, recovered_a1 = recover_line_equation(collinear_pts, gf)
                is_verified = (recovered_a0 == a0_orig) and (recovered_a1 == a1_orig)
            except ValueError:
                is_verified = False
        else:
            is_verified = False

        if is_verified:
            verified_scores.append(count)

    return np.array(verified_scores)


def compute_count_histogram(scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute counts for each collinear points level starting at 2.

    Args:
        scores: Array of max_collinear_count for verified samples

    Returns:
        levels: [2, 3, 4, ...] up to max score
        counts: Count of samples at exactly each level
    """
    if len(scores) == 0:
        return np.array([2]), np.array([0])

    max_score = int(np.max(scores))
    levels = list(range(2, max_score + 1))
    counts = [int(np.sum(scores == t)) for t in levels]

    return np.array(levels), np.array(counts)


def generate_histogram(
    levels: np.ndarray,
    counts: np.ndarray,
    output_path: Path
):
    """
    Generate bar chart showing counts at each collinear points level.

    Args:
        levels: Array of collinear points levels (2, 3, 4, ...)
        counts: Array of counts at each level
        output_path: Path to save PNG
    """
    if len(levels) == 0:
        print("    WARNING: No data to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(levels, counts, color='steelblue', edgecolor='black')

    # Add count labels on top of bars
    max_count = np.max(counts) if len(counts) > 0 else 1
    offset = max_count * 0.02
    for level, count in zip(levels, counts):
        ax.text(level, count + offset, str(count), ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Collinear Points')
    ax.set_ylabel('True Positives')
    ax.set_xticks(levels)
    ax.set_ylim(0, max_count * 1.1)

    # No title, no legend

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> int:
    args = parse_args()
    gf_cache = GFCache()

    # Load experiments
    result = load_and_prepare_experiments(
        min_tokens=args.min_tokens,
        force=args.force,
        input_dir=args.input_dir,
        verbose=args.verbose
    )

    # Create output directory (tp_count_histogram subfolder)
    output_dir = Path(args.output_dir) / "tp_count_histogram"
    output_dir.mkdir(parents=True, exist_ok=True)

    for model, model_data in result['models'].items():
        df = model_data['df']
        config = model_data['config']
        n = config.get('n', 8)

        # Determine workers
        num_workers = args.workers
        if num_workers == 0:
            num_workers = min(cpu_count(), 8)
        workers_info = f", workers={num_workers}" if num_workers > 1 else ""
        print(f"\nModel: {model} (n={n}{workers_info})")

        # Extract samples
        samples = extract_samples(df, config, args.verbose)
        if not samples:
            print("  WARNING: No valid samples, skipping")
            continue

        print(f"  Total samples: {len(samples)}")

        # Run MCP, get verified scores
        t0 = time.perf_counter()
        verified_scores = compute_mcp_verified(samples, gf_cache, args.verbose, args.workers)
        elapsed = time.perf_counter() - t0

        print(f"  Verified: {len(verified_scores)} ({100 * len(verified_scores) / len(samples):.1f}%)")
        print(f"  MCP time: {elapsed:.2f}s")

        if len(verified_scores) == 0:
            print("  WARNING: No verified samples, skipping")
            continue

        # Compute count histogram
        levels, counts = compute_count_histogram(verified_scores)

        # Print counts at each level
        print("  Counts by collinear points:")
        for level, count in zip(levels, counts):
            print(f"    {level}: {count}")

        # Score statistics
        print(f"  Score stats: mean={np.mean(verified_scores):.1f}, "
              f"std={np.std(verified_scores):.1f}, "
              f"min={np.min(verified_scores)}, max={np.max(verified_scores)}")

        # Generate histogram
        model_clean = model.replace('/', '_').replace('\\', '_')
        plot_path = output_dir / f"tp_count_histogram_{model_clean}_n{n}.png"
        generate_histogram(levels, counts, plot_path)
        print(f"  Saved: {plot_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
