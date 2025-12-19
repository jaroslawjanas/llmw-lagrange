"""
Watermark experiment analysis script.

Loads all experiment results, groups by model, and generates statistics and box plots.
"""
import argparse
import json
import shutil
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from lib import load_and_prepare_experiments


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze watermark experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze.py                     # Use per-experiment max_tokens as threshold
  python analyze.py --min-tokens 200    # Global threshold of 200 tokens
  python analyze.py --force             # Proceed despite conflicting parameters
  python analyze.py --input-dir output/my_experiment  # Analyze specific experiment
        """
    )
    parser.add_argument("--min-tokens", type=int, default=None,
                        help="Min tokens filter. Default: use each experiment's max_tokens")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Input directory containing experiment(s). Can be a single experiment folder or parent of multiple. Default: auto-discover from output/")
    parser.add_argument("--output-dir", type=str, default="output/analysis",
                        help="Output directory for results (default: output/analysis)")
    parser.add_argument("--force", action="store_true",
                        help="Proceed even with conflicting parameters")
    return parser.parse_args()


def calculate_stats(df):
    """
    Calculate all statistics for a DataFrame.

    Returns:
        dict with all computed statistics
    """
    stats = {}

    # Key metric
    stats['match_rate'] = df['watermark_recovered'].mean()
    stats['match_rate_std'] = df['watermark_recovered'].std()

    # Block counts from JSON columns
    df['_valid_blocks_count'] = df['valid_blocks'].apply(
        lambda x: len(json.loads(x)) if x else 0
    )
    df['_matching_blocks_count'] = df['matching_blocks'].apply(
        lambda x: len(json.loads(x)) if x else 0
    )

    # Block statistics
    block_cols = {
        'valid_blocks': '_valid_blocks_count',
        'matching_blocks': '_matching_blocks_count',
        'unique_watermark': 'unique_watermark_blocks_count',
        'unique_valid': 'unique_valid_blocks_count',
        'unique_matching': 'unique_matching_blocks_count',
    }

    for name, col in block_cols.items():
        if col in df.columns:
            stats[f'{name}_mean'] = df[col].mean()
            stats[f'{name}_std'] = df[col].std()
            stats[f'{name}_median'] = df[col].median()

    # Token statistics
    stats['properly_encoded_mean'] = df['properly_encoded_tokens'].mean()
    stats['properly_encoded_std'] = df['properly_encoded_tokens'].std()
    stats['properly_encoded_median'] = df['properly_encoded_tokens'].median()

    # Token percentage (properly_encoded / tokens_length)
    df['_properly_encoded_pct'] = df['properly_encoded_tokens'] / df['tokens_length'] * 100
    stats['properly_encoded_pct_mean'] = df['_properly_encoded_pct'].mean()
    stats['properly_encoded_pct_std'] = df['_properly_encoded_pct'].std()
    stats['properly_encoded_pct_median'] = df['_properly_encoded_pct'].median()

    # Timing statistics
    for time_col in ['encoding_time', 'decoding_time', 'mcp_time']:
        if time_col in df.columns:
            stats[f'{time_col}_mean'] = df[time_col].mean()
            stats[f'{time_col}_std'] = df[time_col].std()

    return stats


def create_box_plots(df, output_path, model_name):
    """
    Create 3x4 box plot figure.

    Rows: All, Recovered, Not Recovered
    Cols: unique_watermark_blocks_count, unique_valid_blocks_count, unique_matching_blocks_count, properly_encoded_tokens
    """
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))

    columns = ['unique_watermark_blocks_count', 'unique_valid_blocks_count',
               'unique_matching_blocks_count', 'properly_encoded_tokens']
    col_labels = ['Unique Watermark\nBlocks', 'Unique Valid\nBlocks',
                  'Unique Matching\nBlocks', 'Properly Encoded\nTokens']
    row_labels = ['All', 'Recovered', 'Not Recovered']

    # Subsets
    df_recovered = df[df['watermark_recovered'] == True]
    df_not_recovered = df[df['watermark_recovered'] == False]
    subsets = [df, df_recovered, df_not_recovered]

    for row_idx, (subset, row_label) in enumerate(zip(subsets, row_labels)):
        for col_idx, (col, col_label) in enumerate(zip(columns, col_labels)):
            ax = axes[row_idx, col_idx]

            if col in subset.columns and len(subset) > 0:
                data = subset[col].dropna()
                if len(data) > 0:
                    ax.boxplot(data, vert=True)
                    ax.set_ylabel(col_label if col_idx == 0 else '')
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)

            # Labels
            if row_idx == 0:
                ax.set_title(col_label)
            if col_idx == 0:
                ax.set_ylabel(f'{row_label}\n(n={len(subset)})')

            ax.set_xticks([])

    fig.suptitle(f'Watermark Analysis: {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def format_stats_report(stats, model_name, row_counts, min_tokens_info, source_dirs=None):
    """Format statistics as text report."""
    lines = []
    w = 80  # width

    lines.append('=' * w)
    lines.append('WATERMARK ANALYSIS REPORT')
    lines.append('=' * w)
    lines.append(f'Model: {model_name}')
    lines.append(f'Min tokens: {min_tokens_info}')
    lines.append('')

    # List source directories if multiple experiments were merged
    if source_dirs and len(source_dirs) > 0:
        lines.append('-' * w)
        lines.append(f'SOURCE EXPERIMENTS ({len(source_dirs)})')
        lines.append('-' * w)
        for src in sorted(source_dirs):
            lines.append(f'  {src}')
        lines.append('')

    lines.append('-' * w)
    lines.append('ROW COUNTS')
    lines.append('-' * w)
    lines.append(f"  Total loaded:    {row_counts['total']}")
    lines.append(f"  Included:        {row_counts['included']}")
    lines.append(f"  Excluded:        {row_counts['total'] - row_counts['included']}")
    lines.append('')

    lines.append('-' * w)
    lines.append('KEY METRIC')
    lines.append('-' * w)
    match_rate = stats.get('match_rate', 0)
    lines.append(f"  Match Rate:      {match_rate:.2%}")
    lines.append('')

    lines.append('-' * w)
    lines.append('BLOCK STATISTICS')
    lines.append('-' * w)
    lines.append(f"{'':25} {'Mean':>10} {'Std':>10} {'Median':>10}")

    block_rows = [
        ('Valid blocks', 'valid_blocks'),
        ('Matching blocks', 'matching_blocks'),
        ('Unique watermark', 'unique_watermark'),
        ('Unique valid', 'unique_valid'),
        ('Unique matching', 'unique_matching'),
    ]
    for label, key in block_rows:
        mean = stats.get(f'{key}_mean', 0)
        std = stats.get(f'{key}_std', 0)
        median = stats.get(f'{key}_median', 0)
        lines.append(f"  {label:23} {mean:>10.2f} {std:>10.2f} {median:>10.1f}")
    lines.append('')

    lines.append('-' * w)
    lines.append('TOKEN STATISTICS')
    lines.append('-' * w)
    lines.append(f"{'':25} {'Mean':>10} {'Std':>10} {'Median':>10}")
    lines.append(f"  {'Properly encoded':23} {stats.get('properly_encoded_mean', 0):>10.2f} "
                 f"{stats.get('properly_encoded_std', 0):>10.2f} {stats.get('properly_encoded_median', 0):>10.1f}")
    lines.append(f"  {'Properly encoded (%)':23} {stats.get('properly_encoded_pct_mean', 0):>10.2f} "
                 f"{stats.get('properly_encoded_pct_std', 0):>10.2f} {stats.get('properly_encoded_pct_median', 0):>10.1f}")
    lines.append('')

    lines.append('-' * w)
    lines.append('TIMING (seconds)')
    lines.append('-' * w)
    lines.append(f"{'':25} {'Mean':>10} {'Std':>10}")
    for label, key in [('Encoding', 'encoding_time'), ('Decoding', 'decoding_time'), ('MCP', 'mcp_time')]:
        mean = stats.get(f'{key}_mean', 0)
        std = stats.get(f'{key}_std', 0)
        lines.append(f"  {label:23} {mean:>10.3f} {std:>10.3f}")
    lines.append('')

    lines.append('=' * w)

    return '\n'.join(lines)


def main():
    args = parse_args()

    # Load and prepare experiments using unified loader
    prepared_data = load_and_prepare_experiments(
        min_tokens=args.min_tokens,
        force=args.force,
        input_dir=args.input_dir,
        verbose=True
    )

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy run configs from source experiments
    configs_dir = output_dir / 'source_configs'
    configs_dir.mkdir(exist_ok=True)
    for model_data in prepared_data.values():
        for source_dir in model_data['sources']:
            config_src = Path('output') / source_dir / 'run_config.json'
            if config_src.exists():
                config_dst = configs_dir / f'{source_dir}.json'
                shutil.copy(config_src, config_dst)

    # Process each model group
    for model, model_data in prepared_data.items():
        print(f"\nProcessing model: {model}")

        model_df = model_data['df']
        source_dirs = model_data['sources']
        total_rows = model_data['total_rows']
        included_rows = model_data['included_rows']

        # Calculate stats
        stats = calculate_stats(model_df)
        row_counts = {'total': total_rows, 'included': included_rows}

        # Determine min_tokens info string
        if args.min_tokens is not None:
            min_tokens_info = str(args.min_tokens)
        else:
            min_tokens_info = "per-experiment max_tokens"

        # Generate report
        report = format_stats_report(stats, model, row_counts, min_tokens_info, source_dirs)

        # Save outputs
        model_clean = model.replace('/', '_')
        report_path = output_dir / f"analysis_{model_clean}.txt"
        plot_path = output_dir / f"analysis_{model_clean}.png"

        report_path.write_text(report)
        create_box_plots(model_df, plot_path, model)

        print(f"  Saved: {report_path}")
        print(f"  Saved: {plot_path}")

        # Print report to console
        print()
        print(report)

    print(f"\nAnalysis complete. Results saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
