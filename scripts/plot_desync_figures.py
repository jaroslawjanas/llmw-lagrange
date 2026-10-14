"""
Paper figures for the desynchronization experiments (Appendix: sliding-window extraction).

Figure 1 (front edit): recovery and detection against edit size k when k tokens are
inserted before the first token or the first k tokens are deleted.
Figure 2 (edit position): recovery against the position of a single edit, averaged
over the edit sizes.

Both are written as 300 dpi PNGs sized for one IEEE column (3.5 in).

Inputs are the rate tables written by attack_single_edit.py (rates.csv) or by its
predecessor (rates_by_size.csv, front edit only).

Example (run from the project root):
  python scripts/plot_desync_figures.py \\
      --front-run output/normal/attack_prefix/run_20261008_192518 \\
      --position-run output/normal/attack_single_edit/<run>

Only the edit-position figure, under another name (e.g. for a finer grid of positions):
  python scripts/plot_desync_figures.py --figures position --position-name desync_position_fine \\
      --position-run output/normal/attack_single_edit/<run>
"""
import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COLORS = {'fixed': '#2a78d6', 'sliding': '#eb6834'}
MARKERS = {'fixed': 's', 'sliding': 'o'}
LABELS = {'fixed': 'Fixed blocks', 'sliding': 'Sliding window'}
ATTACKS = ('insertion', 'deletion')

# Matches the paper's other graphs: matplotlib's default sans-serif font, printed at about 5.5-6.5 pt
# once the 3.5 in figure is placed at column width.
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'mathtext.fontset': 'dejavusans',
    'font.size': 6, 'axes.labelsize': 6, 'axes.titlesize': 6.5,
    'xtick.labelsize': 5.5, 'ytick.labelsize': 5.5, 'legend.fontsize': 5.5,
    'lines.linewidth': 1.0, 'lines.markersize': 2.5,
    'axes.linewidth': 0.5, 'grid.linewidth': 0.3,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
})


def parse_args():
    parser = argparse.ArgumentParser(description="Plot the desynchronization figures for the paper")
    parser.add_argument("--front-run", type=str,
                        default=str(PROJECT_ROOT / "output" / "normal" / "attack_prefix" / "run_20261008_192518"),
                        help="Run folder of the front-edit experiment")
    parser.add_argument("--position-run", type=str, default=None,
                        help="Run folder of the edit-position experiment (default: latest in output/normal/attack_single_edit)")
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "paper" / "ACCESS" / "images" / "figures"),
                        help="Where to write the figures (default: paper/ACCESS/images/figures)")
    parser.add_argument("--suffix", type=str, default="n10", help="File name suffix (default: n10)")
    parser.add_argument("--figures", choices=("front", "position", "both"), default="both",
                        help="Which figures to draw (default: both)")
    parser.add_argument("--position-name", type=str, default="desync_position",
                        help="File name stem of the edit-position figure (default: desync_position)")
    return parser.parse_args()


def load_rates(run_dir: str) -> pd.DataFrame:
    run = Path(run_dir)
    path = run / 'rates.csv' if (run / 'rates.csv').exists() else run / 'rates_by_size.csv'
    rates = pd.read_csv(path)
    if 'position' not in rates.columns:
        rates['position'] = 0.0
    return rates


def consecutive_runs(xs: np.ndarray):
    """Split sorted integer sizes into runs of consecutive values (no line across skipped sizes)."""
    breaks = np.where(np.diff(xs) > 1)[0] + 1
    return np.split(np.arange(len(xs)), breaks)


def save(fig, output_dir: Path, name: str) -> None:
    path = output_dir / f'{name}.png'
    fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.02)
    print(f"Saved {path}")
    plt.close(fig)


def plot_front(rates: pd.DataFrame, output_dir: Path, suffix: str) -> None:
    base = rates[rates['attack_type'] == 'none'].iloc[0]
    front = rates[(rates['attack_type'] != 'none') & (rates['position'] == 0.0)]
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.05), sharey=True)
    titles = {'insertion': 'Insert $k$ tokens before the text', 'deletion': 'Delete the first $k$ tokens'}
    for ax, attack in zip(axes, ATTACKS):
        sub = front[front['attack_type'] == attack].sort_values('size')
        ks = sub['size'].to_numpy()
        for skipped in (10, 20):  # multiples of n would re-align the fixed grid
            ax.axvspan(skipped - 0.5, skipped + 0.5, color='0.92', lw=0)
        ax.axhline(base['sliding_recovery'], color=COLORS['sliding'], lw=0.5, alpha=0.6)
        ax.axhline(base['fixed_recovery'], color=COLORS['fixed'], lw=0.5, alpha=0.6)
        for idx in consecutive_runs(ks):
            ax.plot(ks[idx], sub['sliding_recovery'].to_numpy()[idx], color=COLORS['sliding'], marker='o', ms=1.8)
            ax.plot(ks[idx], sub['sliding_detection'].to_numpy()[idx], color=COLORS['sliding'], lw=0.6, marker='o',
                    ms=1.8, mfc='white', mew=0.5)
            ax.plot(ks[idx], sub['fixed_recovery'].to_numpy()[idx], color=COLORS['fixed'], marker='s', ms=1.6)
        ax.set_title(titles[attack], pad=3)
        ax.set_xlim(0, 30)
        ax.set_xticks([1, 5, 10, 15, 20, 25, 29])
        ax.set_ylim(-3, 103)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(True, axis='y', color='0.88')
        ax.set_xlabel('Edit size $k$ (tokens)')
    axes[0].set_ylabel('Rate (%)')
    handles = [
        plt.Line2D([], [], color=COLORS['sliding'], marker='o', ms=2.0, label='Sliding: recovery'),
        plt.Line2D([], [], color=COLORS['sliding'], lw=0.6, marker='o', ms=2.0, mfc='white', mew=0.5,
                   label='Sliding: detection'),
        plt.Line2D([], [], color=COLORS['fixed'], marker='s', ms=2.0, label='Fixed: recovery and detection'),
        plt.Line2D([], [], color='0.45', lw=0.5, label='Same extractor, no attack'),
    ]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
               handlelength=2.2, columnspacing=1.0)
    fig.tight_layout(w_pad=0.6)
    save(fig, output_dir, f'desync_front_{suffix}')


def plot_position(rates: pd.DataFrame, output_dir: Path, suffix: str, stem: str) -> None:
    """Recovery against edit position; each point is the mean over the edit sizes (they differ by a few points)."""
    base = rates[rates['attack_type'] == 'none'].iloc[0]
    attacked = rates[rates['attack_type'] != 'none']
    spread = attacked.groupby(['attack_type', 'position'])[['fixed_recovery', 'sliding_recovery']].agg(lambda v: v.max() - v.min())
    print(f"Largest spread between edit sizes at any position: {spread.to_numpy().max():.2f} points (state it in the caption)")
    positions = 100 * np.sort(attacked['position'].unique())
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.05), sharey=True)
    titles = {'insertion': 'Insertion', 'deletion': 'Deletion'}
    for ax, attack in zip(axes, ATTACKS):
        sub = attacked[attacked['attack_type'] == attack]
        for name in ('fixed', 'sliding'):
            mean = sub.groupby('position')[f'{name}_recovery'].mean()
            ax.axhline(base[f'{name}_recovery'], color=COLORS[name], lw=0.5, alpha=0.6)
            ax.plot(100 * mean.index.to_numpy(), mean.to_numpy(), color=COLORS[name], marker=MARKERS[name], ms=2.2)
        ax.set_title(titles[attack], pad=3)
        # Label every position when there are few; otherwise every 20 % with minor ticks at the positions
        ax.set_xticks(positions if len(positions) <= 6 else np.arange(0, 101, 20))
        if len(positions) > 6:
            ax.set_xticks(positions, minor=True)
        ax.set_xlim(-5, 105)
        ax.set_ylim(-3, 103)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(True, axis='y', color='0.88')
        ax.set_xlabel('Edit position (% of text)')
    axes[0].set_ylabel('Recovery (%)')
    handles = [
        plt.Line2D([], [], color=COLORS['fixed'], marker='s', ms=2.2, label=LABELS['fixed']),
        plt.Line2D([], [], color=COLORS['sliding'], marker='o', ms=2.2, label=LABELS['sliding']),
        plt.Line2D([], [], color='0.45', lw=0.5, label='Same extractor, no attack'),
    ]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
               handlelength=2.0, columnspacing=1.0)
    fig.tight_layout(w_pad=0.6)
    save(fig, output_dir, f'{stem}_{suffix}')


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.figures in ("front", "both"):
        plot_front(load_rates(args.front_run), output_dir, args.suffix)
    if args.figures in ("position", "both"):
        position_run = args.position_run or sorted(glob.glob(str(PROJECT_ROOT / "output" / "normal" / "attack_single_edit" / "run_*")))[-1]
        plot_position(load_rates(position_run), output_dir, args.suffix, args.position_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
