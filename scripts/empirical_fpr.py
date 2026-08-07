#!/usr/bin/env python3
"""
Measure the empirical false-positive rate of MCP-based watermark detection on
unwatermarked human text and compare it with the theoretical union bound.

The detector is run with a fresh random secret key on human-written texts, so
no watermark is embedded. For each field size n and detection threshold tau,
we report the fraction of texts whose maximum collinear count reaches tau.
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# Add project root to path for src imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import src.paths as paths
from src.llm_watermark import LLMWatermarkDecoder, MCPSolver
from src.pm_galois import GaloisField
from src.utils import load_hf_token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Empirical false-positive rate validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python empirical_fpr.py --dataset Hello-SimpleAI/HC3 all train human_answers
  python empirical_fpr.py --dataset ChristophSchuhmann/essays-with-instructions default train instructions \
      --model meta-llama/Llama-2-7b-hf --max-texts 5000
        """
    )
    parser.add_argument("--dataset", type=str, nargs=4, required=True,
                        help="Dataset spec: name subset split column")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf",
                        help="Tokenizer/model name to use for tokenization")
    parser.add_argument("--output-dir", type=str, default="output/empirical_fpr",
                        help="Directory for output CSVs")
    parser.add_argument("--n-values", type=str, default="8,10,12,16",
                        help="Comma-separated field sizes n (default: 8,10,12,16)")
    parser.add_argument("--thresholds", type=str, default="3,4,5,6,7,8",
                        help="Comma-separated detection thresholds tau (default: 3,4,5,6,7,8)")
    parser.add_argument("--min-tokens", type=int, default=208,
                        help="Minimum tokens per text (default: 208)")
    parser.add_argument("--max-texts", type=int, default=None,
                        help="Maximum number of texts to evaluate (default: all that pass filter)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device for decoder ('cuda' or 'cpu'). Default: auto-detect.")
    parser.add_argument("--no-cuda", action="store_true",
                        help="Disable CUDA even if available")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for dataset shuffling and key generation (default: 42)")
    return parser.parse_args()


def parse_int_list(arg: str) -> List[int]:
    values = []
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError(f"No valid integer values in: {arg}")
    return sorted(set(values))


def extract_text(entry: Union[str, List[str], None]) -> Optional[str]:
    """Handle dataset columns that contain a string or a list of strings."""
    if entry is None:
        return None
    if isinstance(entry, list):
        # Take first non-empty answer if column is a list (e.g., HC3 human_answers)
        for item in entry:
            if item:
                return str(item)
        return None
    if isinstance(entry, str):
        return entry if entry.strip() else None
    return str(entry)


def theoretical_fpr(n: int, N: int, tau: int) -> float:
    """Union-bound FPR from Theorem 1: C(N, tau) * 2^{-n(tau-2)}."""
    if tau < 3 or tau > N:
        return 0.0
    return math.comb(N, tau) * (2 ** (-n * (tau - 2)))


def load_texts(dataset_name: str, subset: str, split: str, column: str,
               seed: int, max_texts: Optional[int]) -> List[str]:
    """Load and shuffle human texts from a HuggingFace dataset."""
    token = load_hf_token()
    ds_kwargs = {"cache_dir": paths.DATASETS_CACHE_DIR}
    if token:
        ds_kwargs["token"] = token

    ds = load_dataset(dataset_name, subset, **ds_kwargs)
    if split not in ds:
        raise ValueError(f"Split '{split}' not found. Available: {list(ds.keys())}")

    data = ds[split]
    all_texts = []
    print(f"  Extracting texts from {len(data)} rows...", flush=True)
    for row in tqdm(data, desc="Extracting texts", mininterval=2.0):
        text = extract_text(row.get(column))
        if text:
            all_texts.append(text)

    print(f"  Extracted {len(all_texts)} non-empty texts; shuffling...", flush=True)
    rng = random.Random(seed)
    rng.shuffle(all_texts)

    if max_texts is not None and len(all_texts) > max_texts:
        all_texts = all_texts[:max_texts]

    return all_texts


def main() -> int:
    args = parse_args()
    n_values = parse_int_list(args.n_values)
    thresholds = parse_int_list(args.thresholds)

    # Resolve device
    if args.device:
        device = args.device
    elif args.no_cuda:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset_name, subset, split, column = args.dataset
    print(f"Loading texts: {dataset_name}/{subset} split={split} column={column}", flush=True)
    texts = load_texts(dataset_name, subset, split, column, args.seed, args.max_texts)
    print(f"Loaded {len(texts)} candidate texts", flush=True)

    # Load tokenizer
    token = load_hf_token()
    tok_kwargs = {"cache_dir": paths.MODELS_CACHE_DIR}
    if token:
        tok_kwargs["token"] = token
    tokenizer = AutoTokenizer.from_pretrained(args.model, **tok_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer loaded: {args.model}", flush=True)

    # Tokenize and filter by length
    tokenized_texts = []
    kept = 0
    for i, text in enumerate(tqdm(texts, desc="Tokenizing/filtering", mininterval=2.0)):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) >= args.min_tokens:
            tokenized_texts.append(ids[:args.min_tokens])
            kept += 1
        if (i + 1) % 1000 == 0:
            print(f"  ...processed {i + 1}/{len(texts)}, kept {kept}", flush=True)

    print(f"Kept {len(tokenized_texts)} texts with >= {args.min_tokens} tokens", flush=True)
    if not tokenized_texts:
        print("ERROR: No texts satisfied the min-tokens filter.")
        return 1

    # Determine N = floor(min_tokens / n) per n value
    n_to_N = {n: args.min_tokens // n for n in n_values}

    records = []
    summary_records = []

    for n in n_values:
        N = n_to_N[n]
        print(f"\nEvaluating n={n} (N={n_to_N[n]} blocks per text)", flush=True)

        # Use a fresh random secret key so the text is effectively unwatermarked
        rng_key = random.Random(args.seed + n)
        secret_key_int = rng_key.getrandbits(64)
        secret_key = f"{secret_key_int:016x}"

        gf = GaloisField(n)
        decoder = LLMWatermarkDecoder(
            model_name=args.model,
            secret_key=secret_key,
            n=n,
            gf=gf,
            device=device,
            hamming_mode="none",
            correct=False,
            c_correction=0,
            verbose=False
        )
        mcp_solver = MCPSolver(gf=gf, n=n, verbose=False)

        mcp_counts = []
        print(f"  Running MCP on {len(tokenized_texts)} texts...", flush=True)
        for i, token_ids in enumerate(tqdm(tokenized_texts, desc=f"n={n}", mininterval=2.0)):
            all_blocks, _ = decoder._decode_fixed_blocks(token_ids)
            points = [(b["x"], b["y"]) for b in all_blocks]
            max_count, _, _ = mcp_solver.solve_mcp(points)
            mcp_counts.append(max_count)

            records.append({
                "n": n,
                "N": N,
                "text_idx": i,
                "mcp_count": max_count,
                "num_points": len(points)
            })

            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(tokenized_texts)} texts", flush=True)

        # Compute empirical FPR per threshold
        mcp_counts = np.array(mcp_counts)
        for tau in thresholds:
            empirical = np.mean(mcp_counts >= tau)
            theoretical = theoretical_fpr(n, N, tau)
            summary_records.append({
                "n": n,
                "N": N,
                "tau": tau,
                "num_texts": len(tokenized_texts),
                "empirical_fpr": empirical,
                "theoretical_fpr": theoretical,
                "mcp_count_max": int(mcp_counts.max()),
                "mcp_count_mean": float(mcp_counts.mean()),
                "mcp_count_std": float(mcp_counts.std()),
            })

    # Save outputs
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    per_text_file = output_path / f"fpr_per_text_{timestamp}.csv"
    pd.DataFrame(records).to_csv(per_text_file, index=False)
    print(f"\nSaved per-text counts to: {per_text_file}", flush=True)

    summary_df = pd.DataFrame(summary_records)
    summary_file = output_path / f"fpr_summary_{timestamp}.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Saved summary to: {summary_file}", flush=True)

    # Print the Markdown-style table to console as soon as data is ready
    print("\n" + "=" * 80, flush=True)
    print("Empirical vs. theoretical FPR", flush=True)
    print("=" * 80, flush=True)
    print(f"{'n':>3} {'N':>3} {'tau':>4} {'empirical':>12} {'theoretical':>12} {'ratio':>8}", flush=True)
    for _, row in summary_df.iterrows():
        emp = row["empirical_fpr"]
        theo = row["theoretical_fpr"]
        ratio = emp / theo if theo > 0 else float("inf")
        print(f"{int(row['n']):>3} {int(row['N']):>3} {int(row['tau']):>4} "
              f"{emp:>12.4e} {theo:>12.4e} {ratio:>8.3f}", flush=True)

    # Build human-readable report.txt
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("Empirical False-Positive Rate Validation")
    report_lines.append("=" * 80)
    report_lines.append(f"Dataset: {dataset_name}/{subset} split={split} column={column}")
    report_lines.append(f"Tokenizer: {args.model}")
    report_lines.append(f"Texts evaluated: {len(tokenized_texts)} (out of {len(texts)} candidate texts)")
    report_lines.append(f"Minimum token count: {args.min_tokens}")
    report_lines.append(f"Random seed: {args.seed}")
    report_lines.append(f"Secret keys: deterministically random per n (no watermark embedded)")
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("Empirical vs. Theoretical FPR")
    report_lines.append("=" * 80)
    report_lines.append(f"{'n':>3} {'N':>3} {'tau':>4} {'empirical':>14} {'theoretical':>14} {'ratio':>8}")
    report_lines.append("-" * 80)
    for _, row in summary_df.iterrows():
        emp = row["empirical_fpr"]
        theo = row["theoretical_fpr"]
        ratio = emp / theo if theo > 0 else float("inf")
        report_lines.append(
            f"{int(row['n']):>3} {int(row['N']):>3} {int(row['tau']):>4} "
            f"{emp:>14.4e} {theo:>14.4e} {ratio:>8.3f}"
        )

    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("Notes")
    report_lines.append("=" * 80)
    report_lines.append("The theoretical FPR is the union-bound upper bound from Theorem 1:")
    report_lines.append("  FPR <= C(N, tau) * 2^(-n(tau-2))")
    report_lines.append("The empirical FPR is the fraction of unwatermarked human texts whose")
    report_lines.append("maximum collinear point count reaches or exceeds the threshold tau.")
    report_lines.append("")

    report_file = output_path / f"fpr_report_{timestamp}.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"Saved report to: {report_file}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
