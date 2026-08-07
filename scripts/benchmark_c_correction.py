#!/usr/bin/env python3
"""
Benchmark c-correction runtime overhead for watermark extraction.

Loads an existing experiment (statistics.parquet + run_config.json), re-uses the
stored generated token IDs, and re-runs the decoder + MCP solver with different
c-correction radii. Reports per-phase timing (decode, candidate generation, MCP)
and exact recovery rates.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Add project root to path for src imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_watermark import LLMWatermarkDecoder, MCPSolver
from src.pm_galois import GaloisField
from scripts.lib import ExperimentLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark c-correction runtime overhead",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_c_correction.py --input-dir output/c-correction/exp_name
  python benchmark_c_correction.py --input-dir output/normal/exp_name --c-values 0,1,2,3
  python benchmark_c_correction.py --input-dir output --model meta-llama/Llama-2-7b-hf --n 16
        """
    )
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Experiment directory (contains statistics.parquet and run_config.json) "
                             "or a parent directory of experiments. Required.")
    parser.add_argument("--output-dir", type=str, default="output/benchmark_c_correction",
                        help="Directory for the timing CSV output (default: output/benchmark_c_correction)")
    parser.add_argument("--model", type=str, default=None,
                        help="Filter experiments by model name (optional)")
    parser.add_argument("--n", type=int, default=None,
                        help="Filter experiments by field size n (optional)")
    parser.add_argument("--c-values", type=str, default="0,1,2",
                        help="Comma-separated c-correction radii to benchmark (default: 0,1,2)")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Maximum number of rows to benchmark (default: all rows)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device for decoder ('cuda' or 'cpu'). Default: auto-detect.")
    parser.add_argument("--no-cuda", action="store_true",
                        help="Disable CUDA even if available")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for row shuffling (default: 42)")
    return parser.parse_args()


def parse_c_values(c_values_arg: str) -> List[int]:
    values = []
    for part in c_values_arg.split(","):
        part = part.strip()
        if not part:
            continue
        val = int(part)
        if val < 0:
            raise ValueError(f"c-correction radius must be >= 0, got {val}")
        values.append(val)
    if not values:
        raise ValueError("No valid c-values provided")
    return sorted(set(values))


def discover_experiment(input_dir: Path, model: Optional[str], n: Optional[int]) -> Path:
    """Resolve input_dir to a single experiment directory matching model/n."""
    input_path = Path(input_dir)
    candidates = []

    # If input_dir itself contains statistics.parquet, use it directly
    if (input_path / "statistics.parquet").exists():
        candidates.append(input_path)
    else:
        # Otherwise treat as parent directory
        loader = ExperimentLoader(output_dir=input_path)
        for exp_dir in loader._discover_experiments():
            config = loader._load_config(exp_dir)
            if config is None:
                continue
            if model is not None and config.get("model") != model:
                continue
            if n is not None and config.get("n") != n:
                continue
            candidates.append(exp_dir)

    if not candidates:
        raise FileNotFoundError(f"No experiment found in {input_dir} matching model={model}, n={n}")
    if len(candidates) > 1:
        print(f"WARNING: Multiple experiments matched; using first:")
        for c in candidates:
            print(f"  {c.name}")
    return candidates[0]


def benchmark_row(
    generated_ids: List[int],
    decoder: LLMWatermarkDecoder,
    mcp_solver: MCPSolver,
    original_a0: int,
    original_a1: int,
    c: int
) -> Dict:
    """Run one row through decode + c-correction + MCP and return timings."""
    # Make sure the decoder is using the requested c value
    decoder.c_correction = c

    # Phase 1: fixed-block decode (common to all c values)
    decode_start = time.perf_counter()
    all_blocks, _ = decoder._decode_fixed_blocks(generated_ids)
    decode_time = time.perf_counter() - decode_start
    num_original = len(all_blocks)

    # Phase 2: c-correction expansion
    variation_start = time.perf_counter()
    if c > 0:
        expanded_blocks = decoder._apply_c_correction(all_blocks)
    else:
        expanded_blocks = all_blocks
    variation_time = time.perf_counter() - variation_start
    num_expanded = len(expanded_blocks)

    # Phase 3: MCP verification
    mcp_start = time.perf_counter()
    result = mcp_solver.verify_watermark(expanded_blocks, original_a0, original_a1)
    mcp_time = time.perf_counter() - mcp_start

    return {
        "c": c,
        "num_original_blocks": num_original,
        "num_expanded_blocks": num_expanded,
        "decode_ms": decode_time * 1000.0,
        "variation_ms": variation_time * 1000.0,
        "mcp_ms": mcp_time * 1000.0,
        "total_ms": (decode_time + variation_time + mcp_time) * 1000.0,
        "recovered": bool(result.get("is_valid", False)),
        "max_collinear_count": int(result.get("max_collinear_count", 0)),
    }


def main() -> int:
    args = parse_args()
    c_values = parse_c_values(args.c_values)

    # Resolve device
    if args.device:
        device = args.device
    elif args.no_cuda:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)

    # Locate experiment
    try:
        exp_dir = discover_experiment(Path(args.input_dir), args.model, args.n)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1
    print(f"Benchmarking experiment: {exp_dir}", flush=True)

    # Load config and data
    config_path = exp_dir / "run_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    df = pd.read_parquet(exp_dir / "statistics.parquet")
    print(f"Loaded {len(df)} rows", flush=True)

    # Extract parameters from config
    model_name = config["model"]
    n = config["n"]
    green_fraction = config.get("green_fraction", 0.5)
    hamming_mode = config.get("hamming", "none")
    correct = config.get("correct", False)

    # Secret key is stored per row in the statistics file
    secret_key = str(df.iloc[0]["secret_key"])

    # Optionally subsample rows deterministically
    if args.max_rows is not None and args.max_rows < len(df):
        rng = random.Random(args.seed)
        indices = rng.sample(range(len(df)), args.max_rows)
        df = df.iloc[indices].reset_index(drop=True)
        print(f"Subsampled to {len(df)} rows (seed={args.seed})")

    gf = GaloisField(n)

    records = []
    print(f"Benchmarking {len(df)} rows for c in {c_values}...", flush=True)

    # Re-use a single decoder and MCP solver across rows. The decoder only needs
    # the same model/key parameters as the original experiment; tokenizer loading
    # is the expensive step we want to avoid repeating.
    decoder = LLMWatermarkDecoder(
        model_name=model_name,
        secret_key=secret_key,
        n=n,
        gf=gf,
        device=device,
        green_list_fraction=green_fraction,
        hamming_mode=hamming_mode,
        correct=correct,
        c_correction=0,  # overridden per benchmark
        verbose=False
    )
    mcp_solver = MCPSolver(gf=gf, n=n, verbose=False)

    for row_idx, row in tqdm(df.iterrows(), total=len(df), desc="Benchmarking rows", mininterval=2.0):
        generated_ids = json.loads(row["generated_ids"])
        original_a0 = int(row["a0"])
        original_a1 = int(row["a1"])

        for c in c_values:
            record = benchmark_row(generated_ids, decoder, mcp_solver,
                                   original_a0, original_a1, c)
            record["row_idx"] = int(row_idx)
            records.append(record)

        if (row_idx + 1) % 100 == 0:
            print(f"  Processed {row_idx + 1}/{len(df)} rows", flush=True)

    # Save results
    results_df = pd.DataFrame(records)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_file = output_path / f"c_correction_timing_n{n}_{timestamp}.csv"
    results_df.to_csv(out_file, index=False)
    print(f"\nSaved per-row timings to: {out_file}", flush=True)

    # Print aggregate summary
    print("\n" + "=" * 70, flush=True)
    print("Aggregate timing summary (milliseconds)", flush=True)
    print("=" * 70, flush=True)
    summary = results_df.groupby("c").agg(
        rows=("row_idx", "nunique"),
        recovery_rate=("recovered", "mean"),
        original_blocks=("num_original_blocks", "mean"),
        expanded_blocks=("num_expanded_blocks", "mean"),
        decode_mean=("decode_ms", "mean"),
        decode_std=("decode_ms", "std"),
        variation_mean=("variation_ms", "mean"),
        variation_std=("variation_ms", "std"),
        mcp_mean=("mcp_ms", "mean"),
        mcp_std=("mcp_ms", "std"),
        total_mean=("total_ms", "mean"),
        total_std=("total_ms", "std"),
    ).reset_index()
    print(summary.to_string(index=False))

    # Save summary too
    summary_file = output_path / f"c_correction_summary_n{n}_{timestamp}.csv"
    summary.to_csv(summary_file, index=False)
    print(f"\nSaved summary to: {summary_file}", flush=True)

    # Build human-readable report
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("C-Correction Runtime Benchmark")
    report_lines.append("=" * 70)
    report_lines.append(f"Experiment: {exp_dir.name}")
    report_lines.append(f"Model: {model_name}")
    report_lines.append(f"Field size n: {n}")
    report_lines.append(f"Device: {device}")
    report_lines.append(f"Rows benchmarked: {len(df)}")
    report_lines.append(f"C-values: {c_values}")
    report_lines.append("")
    report_lines.append("=" * 70)
    report_lines.append("Aggregate timing summary (milliseconds)")
    report_lines.append("=" * 70)
    report_lines.append(
        f"{'c':>3} {'rows':>6} {'rec%':>7} {'orig_blocks':>12} {'exp_blocks':>12} "
        f"{'decode_ms':>12} {'var_ms':>12} {'mcp_ms':>12} {'total_ms':>12}"
    )
    report_lines.append("-" * 70)
    for _, row in summary.iterrows():
        report_lines.append(
            f"{int(row['c']):>3} {int(row['rows']):>6} {row['recovery_rate']*100:>6.1f}% "
            f"{row['original_blocks']:>11.1f} {row['expanded_blocks']:>11.1f} "
            f"{row['decode_mean']:>11.2f} {row['variation_mean']:>11.2f} "
            f"{row['mcp_mean']:>11.2f} {row['total_mean']:>11.2f}"
        )
    report_lines.append("")
    report_lines.append("All times are means over the benchmarked rows, in milliseconds.")
    report_lines.append("decode_ms = fixed-block decoding; var_ms = c-correction candidate generation;")
    report_lines.append("mcp_ms = MCP solve + line recovery; total_ms = sum of the three phases.")

    report_file = output_path / f"c_correction_report_n{n}_{timestamp}.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"Saved report to: {report_file}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
