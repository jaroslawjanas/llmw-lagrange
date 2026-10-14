"""
Single-edit desynchronization attack.

Inserts k random tokens, or deletes k tokens, at a chosen position of every
watermarked text (as a fraction of its length: 0 = before the first token,
1 = after the last). The edit size k is never a multiple of the block width n,
so every fixed block after the edit stays shifted by k mod n and the fixed-block
grid can never fall back into alignment. Each attacked text (and the unattacked
baseline) is decoded with fixed-block and sliding-window extraction from the
same bit stream.

Positions: an insertion goes before token round(f * L) (f = 1 appends); a
deletion removes tokens [p, p + k) with p = min(round(f * L), L - k), so f = 1
removes the last k tokens.

Reuses the decoder cache, fast MCP solver and detection threshold from
attack_simulation.py.

Examples (run from the project root):
  # Edits before the first block, k = 1 .. 3n-1 (no multiples of n)
  python scripts/attack_single_edit.py --input-dir output/normal/<experiment> -j 8
  # One edit of 1, 5, 15 or 25 tokens at 0%, 25%, 50%, 75% and 100% of the text
  python scripts/attack_single_edit.py --input-dir output/normal/<experiment> \\
      --positions 0,0.25,0.5,0.75,1 --sizes 1,5,15,25 -j 8
"""
import argparse
import json
import multiprocessing as mp
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import src.paths as paths
from attack_simulation import (DEFAULT_ALPHA, SimulationCache, check_device_consistency,
                               detection_tau, get_device)
from lib import load_and_prepare_experiments

EXTRACTORS = ('fixed', 'sliding')
ATTACKS = ('insertion', 'deletion')


def parse_args():
    parser = argparse.ArgumentParser(
        description="Insert or delete tokens at chosen positions and compare fixed vs sliding extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Experiment folder (or parent of experiments) to attack; plain (non-Hamming) only")
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "output" / "attacks_single_edit"),
                        help="Output directory for results (default: output/attacks_single_edit)")
    parser.add_argument("--positions", type=str, default="0",
                        help="Edit positions as fractions of the text length, e.g. '0,0.25,0.5,0.75,1' "
                             "(default: 0, before the first token)")
    parser.add_argument("--sizes", type=str, default=None,
                        help="Edit sizes k, e.g. '1-29' or '1,3,7'. Multiples of n are rejected. "
                             "Default: 1 .. 3n-1 without multiples of n")
    parser.add_argument("--min-tokens", type=int, default=None,
                        help="Min tokens filter. Default: use each experiment's max_tokens")
    parser.add_argument("--seed", type=int, default=42, help="Base seed; row seed = seed + row index (default: 42)")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                        help="False-positive target for the detection threshold tau (default: 0.01)")
    parser.add_argument("--limit", type=int, default=None, help="Only attack the first N rows (for quick checks)")
    parser.add_argument("--force", action="store_true", help="Proceed even with conflicting parameters")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA even if available")
    parser.add_argument("-j", "--workers", type=int, default=0,
                        help="Number of worker processes (0=auto, 1=serial, default: 0)")
    return parser.parse_args()


def parse_sizes(arg: Optional[str], n: int) -> List[int]:
    """Parse '1-29' / '1,3,7' into sorted sizes; reject multiples of n (they re-align the fixed grid)."""
    if arg is None:
        return [k for k in range(1, 3 * n) if k % n]
    sizes = set()
    for part in arg.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            lo, hi = (int(v) for v in part.split('-', 1))
            sizes.update(range(lo, hi + 1))
        else:
            sizes.add(int(part))
    synced = sorted(k for k in sizes if k % n == 0)
    if synced:
        raise ValueError(f"sizes {synced} are multiples of n = {n}: the fixed-block grid would re-align. "
                         f"Use sizes that are not multiples of n.")
    if not sizes or min(sizes) < 1:
        raise ValueError("sizes must be positive integers")
    return sorted(sizes)


def parse_positions(arg: str) -> List[float]:
    """Parse '0,0.25,1' into fractions of the text length (kept in the given order)."""
    positions = [float(p) for p in arg.split(',') if p.strip()]
    if not positions or any(not 0.0 <= p <= 1.0 for p in positions):
        raise ValueError("positions must be fractions between 0 and 1")
    if len(set(positions)) != len(positions):
        raise ValueError("positions must be distinct")
    return positions


# =============================================================================
# Per-row simulation (worker side)
# =============================================================================

_worker: Dict[str, Any] = {}


def _init_worker(device: str, cache_dir: str) -> None:
    paths.set_cache_dir(cache_dir)
    _worker['cache'] = SimulationCache()
    _worker['device'] = device


def evaluate(decoder, mcp_solver, gf, token_ids: List[int], a0: int, a1: int,
             watermark_blocks: List[Dict], n: int, alpha: float, timing: Dict[str, float]) -> Dict[str, Any]:
    """Decode token_ids with both extractors and verify against the embedded line."""
    t0 = time.perf_counter()
    decoded = decoder.decode_with_extractors(token_ids, EXTRACTORS)
    timing['decode'] += time.perf_counter() - t0
    out = {'text_len': len(token_ids)}
    for name in EXTRACTORS:
        blocks = decoded[name]
        t0 = time.perf_counter()
        verification = mcp_solver.verify_watermark(blocks, a0, a1, watermark_blocks)
        timing['mcp'] += time.perf_counter() - t0
        points = set((b['x'], b['y']) for b in blocks)
        tau = detection_tau(len(blocks), n, alpha)
        out[f'{name}_recovered'] = verification['is_valid']
        out[f'{name}_detected'] = verification['is_valid'] and verification['max_collinear_count'] >= tau
        out[f'{name}_max_collinear'] = verification['max_collinear_count']
        out[f'{name}_tau'] = tau
        out[f'{name}_points'] = len(blocks)
        out[f'{name}_true_points'] = sum(1 for x, y in points if y == gf.add(a0, gf.multiply(a1, x)))
    return out


def _process_row(task: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    cache, device, config = _worker['cache'], _worker['device'], task['config']
    timing = {'decode': 0.0, 'mcp': 0.0}
    n = config['n']
    gf = cache.get_gf(n)
    mcp_solver = cache.get_mcp_solver(n)
    decoder = cache.get_decoder(model_name=config['model'], secret_key=task['secret_key'], n=n, gf=gf,
                                hamming_mode='none', correct=False, green_fraction=config['green_fraction'],
                                device=device)
    ids = json.loads(task['generated_ids'])
    a0, a1 = task['a0'], task['a1']
    watermark_blocks = json.loads(task['watermark_blocks'])
    rng = random.Random(task['seed'])
    meta = {'row_idx': task['row_idx'], 'original_len': len(ids)}

    def run(attack_type: str, position: float, start: int, size: int, text: List[int]) -> Dict[str, Any]:
        return {**meta, 'attack_type': attack_type, 'position': position, 'edit_start': start, 'size': size,
                **evaluate(decoder, mcp_solver, gf, text, a0, a1, watermark_blocks, n, task['alpha'], timing)}

    L = len(ids)
    results = [run('none', 0.0, 0, 0, ids)]
    # Loop order (position, size, insertion then deletion) keeps the RNG draws of a single-position
    # run identical to the original front-edit script.
    for position in task['positions']:
        for k in task['sizes']:
            # Insertion: k random tokens before token round(position * L) (same distribution as attack_simulation)
            start = round(position * L)
            inserted = [rng.randint(0, decoder.vocab_size - 1) for _ in range(k)]
            results.append(run('insertion', position, start, k, ids[:start] + inserted + ids[start:]))
            # Deletion: k tokens from round(position * L), shifted back so the last k tokens are the latest option
            if L - k >= n:
                start = min(round(position * L), L - k)
                results.append(run('deletion', position, start, k, ids[:start] + ids[start + k:]))
    return results, timing


# =============================================================================
# Reporting
# =============================================================================

def rate_table(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (attack_type, position, size), grp in results.groupby(['attack_type', 'position', 'size']):
        rec = {'attack_type': attack_type, 'position': float(position), 'size': int(size), 'texts': len(grp)}
        for name in EXTRACTORS:
            rec[f'{name}_recovery'] = 100 * grp[f'{name}_recovered'].mean()
            rec[f'{name}_detection'] = 100 * grp[f'{name}_detected'].mean()
            rec[f'{name}_true_points'] = grp[f'{name}_true_points'].mean()
            rec[f'{name}_points'] = grp[f'{name}_points'].mean()
            rec[f'{name}_tau'] = grp[f'{name}_tau'].mean()
        rows.append(rec)
    order = {'none': 0, 'insertion': 1, 'deletion': 2}
    rates = pd.DataFrame(rows)
    rates['_order'] = rates['attack_type'].map(order)
    return rates.sort_values(['_order', 'position', 'size']).drop(columns='_order').reset_index(drop=True)


def format_summary(rates: pd.DataFrame, args, n: int, sources: List[str], elapsed: float, timing: Dict[str, float]) -> str:
    w = 92
    base = rates[rates['attack_type'] == 'none'].iloc[0]
    lines = ['=' * w, 'SINGLE-EDIT DESYNCHRONIZATION ATTACK: FIXED-BLOCK vs SLIDING-WINDOW EXTRACTION', '=' * w,
             f'Edit: insert k random tokens or delete k tokens at a fraction of the text length '
             f'(k never a multiple of n = {n})',
             f'Texts: {int(base["texts"])}   Seed: {args.seed}   Detection alpha: {args.alpha:g}   Elapsed: {elapsed:.1f}s',
             f'Aggregate CPU time: decode {timing["decode"]:.1f}s, mcp {timing["mcp"]:.1f}s',
             '', 'SOURCE EXPERIMENTS'] + [f'  {s}' for s in sorted(sources)]
    lines += ['', f'No attack: recovery fixed {base["fixed_recovery"]:.1f}% / sliding {base["sliding_recovery"]:.1f}%, '
                  f'detection fixed {base["fixed_detection"]:.1f}% / sliding {base["sliding_detection"]:.1f}%']
    attacked = rates[rates['attack_type'] != 'none']
    for position in sorted(attacked['position'].unique()):
        at_pos = attacked[attacked['position'] == position]
        for metric, title in (('recovery', 'RECOVERY RATE (%): recovered line equals the embedded line'),
                              ('detection', f'DETECTION RATE (%): recovered and max collinear count >= tau (alpha={args.alpha:g})'),
                              ('true_points', 'WATERMARK POINTS PER TEXT (mean, distinct points on the embedded line)')):
            lines += ['', '-' * w, f'EDIT AT {100 * position:g}% OF THE TEXT - {title}', '-' * w,
                      f"{'k':>4} | {'Insertion':^26} | {'Deletion':^26}",
                      f"{'':>4} | {'fixed':>8} {'sliding':>8} {'diff':>8} | {'fixed':>8} {'sliding':>8} {'diff':>8}"]
            for k in sorted(at_pos['size'].unique()):
                row = f'{k:>4}'
                for attack_type in ATTACKS:
                    r = at_pos[(at_pos['attack_type'] == attack_type) & (at_pos['size'] == k)]
                    if r.empty:
                        row += f" | {'':>26}"
                        continue
                    f, s = r.iloc[0][f'fixed_{metric}'], r.iloc[0][f'sliding_{metric}']
                    fmt = '{:>8.2f}' if metric == 'true_points' else '{:>7.1f}%'
                    row += f" | {fmt.format(f)} {fmt.format(s)} {s - f:>+8.1f}"
                lines.append(row)
    lines += ['', '=' * w]
    return '\n'.join(lines)


def plot_rates(rates: pd.DataFrame, metric: str, ylabel: str, output_path: Path) -> None:
    """One panel per edit type: rate against edit size (single position) or against position (several)."""
    colors = {'fixed': 'tab:blue', 'sliding': 'tab:orange'}
    labels = {'fixed': 'Fixed blocks', 'sliding': 'Sliding window'}
    base = rates[rates['attack_type'] == 'none'].iloc[0]
    attacked = rates[rates['attack_type'] != 'none']
    by_position = attacked['position'].nunique() > 1
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, attack_type in zip(axes, ATTACKS):
        sub = attacked[attacked['attack_type'] == attack_type]
        for name in EXTRACTORS:
            if by_position:
                curve = sub.groupby('position')[f'{name}_{metric}'].mean()
                x = 100 * curve.index.to_numpy()
            else:
                curve = sub.set_index('size')[f'{name}_{metric}'].sort_index()
                x = curve.index.to_numpy()
            ax.plot(x, curve.to_numpy(), marker='o', linewidth=2, markersize=4, color=colors[name], label=labels[name])
            ax.axhline(base[f'{name}_{metric}'], color=colors[name], linewidth=1, alpha=0.4)
        if by_position:
            sizes = ', '.join(str(k) for k in sorted(sub['size'].unique()))
            ax.set_title(f'{attack_type.capitalize()} (mean over k = {sizes})', fontsize=12, fontweight='bold')
            ax.set_xlabel('Edit position (% of the text)', fontsize=11)
            ax.set_xticks(100 * sorted(sub['position'].unique()))
        else:
            ax.set_title(f'{attack_type.capitalize()} of k tokens', fontsize=12, fontweight='bold')
            ax.set_xlabel('Edit size k (tokens; multiples of n excluded)', fontsize=11)
            ax.set_xticks(sorted(sub['size'].unique()))
            ax.tick_params(axis='x', labelsize=8)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(ylabel, fontsize=11)
    axes[0].legend(loc='best')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved graph: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    args = parse_args()
    paths.set_cache_dir(str(PROJECT_ROOT / "cache"))
    device = get_device(args.no_cuda)
    num_workers = args.workers or min(mp.cpu_count(), 8)

    prepared = load_and_prepare_experiments(min_tokens=args.min_tokens, force=args.force,
                                            input_dir=args.input_dir, verbose=True)
    models = prepared['models']
    if len(models) != 1:
        print(f"Error: expected one model, found {list(models)}")
        return 1
    check_device_consistency(models, device)
    model, model_data = next(iter(models.items()))
    config, df = model_data['config'], model_data['df']
    if config.get('hamming', 'none') != 'none':
        print("Error: this attack compares plain extractors; the experiment uses Hamming coding.")
        return 1
    n = int(config['n'])
    try:
        sizes = parse_sizes(args.sizes, n)
        positions = parse_positions(args.positions)
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    if args.limit is not None:
        df = df.head(args.limit)
    print(f"Device: {device}   Workers: {num_workers}   n = {n}   Positions: {positions}   Sizes: {sizes}")

    tasks = [{
        'row_idx': idx, 'seed': args.seed + idx, 'sizes': sizes, 'positions': positions, 'alpha': args.alpha,
        'config': config, 'generated_ids': row['generated_ids'], 'secret_key': row['secret_key'],
        'a0': int(row['a0']), 'a1': int(row['a1']), 'watermark_blocks': row['watermark_blocks'],
    } for idx, row in df.iterrows()]

    all_rows: List[Dict[str, Any]] = []
    timing = {'decode': 0.0, 'mcp': 0.0}
    t_start = time.perf_counter()
    if num_workers == 1:
        _init_worker(device, paths.CACHE_DIR)
        for res, t in tqdm(map(_process_row, tasks), total=len(tasks), desc="Texts"):
            all_rows.extend(res)
            for key in timing:
                timing[key] += t[key]
    else:
        ctx = mp.get_context('spawn')  # CUDA-safe on every platform
        with ctx.Pool(processes=num_workers, initializer=_init_worker, initargs=(device, paths.CACHE_DIR)) as pool:
            for res, t in tqdm(pool.imap_unordered(_process_row, tasks, chunksize=4), total=len(tasks), desc="Texts"):
                all_rows.extend(res)
                for key in timing:
                    timing[key] += t[key]
    elapsed = time.perf_counter() - t_start

    results = pd.DataFrame(all_rows).sort_values(['row_idx', 'attack_type', 'position', 'size']).reset_index(drop=True)
    rates = rate_table(results)

    output_dir = Path(args.output_dir) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    (output_dir / 'source_configs').mkdir(parents=True, exist_ok=True)
    for source in model_data['sources']:
        src = prepared['base_path'] / source / 'run_config.json'
        if src.exists():
            shutil.copy(src, output_dir / 'source_configs' / f'{source}.json')
    results.to_csv(output_dir / 'attack_results.csv', index=False)
    rates.to_csv(output_dir / 'rates.csv', index=False)
    plot_rates(rates, 'recovery', 'Recovery Rate (%)', output_dir / 'recovery.png')
    plot_rates(rates, 'detection', f'Detection Rate (%), alpha = {args.alpha:g}', output_dir / 'detection.png')
    plot_rates(rates, 'true_points', 'Watermark points per text (mean)', output_dir / 'true_points.png')
    summary = format_summary(rates, args, n, model_data['sources'], elapsed, timing)
    (output_dir / 'attack_summary.txt').write_text(summary, encoding='utf-8')
    print('\n' + summary)
    print(f"\nResults saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
