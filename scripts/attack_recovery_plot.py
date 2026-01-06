#!/usr/bin/env python3
"""Plot attack recovery rates from one or two CSVs."""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Plot attack recovery rates from one or two CSVs')
    parser.add_argument('--csv1', type=str, required=True, help='First recovery_rate.csv')
    parser.add_argument('--csv2', type=str, default=None, help='Second recovery_rate.csv (optional)')
    parser.add_argument('--output', type=str, default='attack_recovery.png', help='Output PNG path')
    args = parser.parse_args()

    # Read CSVs
    df1 = pd.read_csv(args.csv1)
    dataframes = [df1]
    if args.csv2:
        df2 = pd.read_csv(args.csv2)
        dataframes.append(df2)

    # Color schemes
    colors = [
        {'curve': 'tab:blue', 'baseline': 'green'},
        {'curve': 'tab:orange', 'baseline': 'tab:red'},
    ]

    attack_types = ['insertion', 'deletion', 'substitution']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, attack_type in zip(axes, attack_types):
        all_groups = []
        for df, color in zip(dataframes, colors):
            attack_data = df[df['attack_type'] == attack_type]
            groups = attack_data['groups'].tolist()
            rates = attack_data['recovery_rate'].tolist()
            all_groups = groups  # Use last one for x-ticks

            # Plot curve
            ax.plot(groups, rates, marker='o', linewidth=2, markersize=6,
                    color=color['curve'])

            # Plot baseline
            if 0 in groups:
                baseline_rate = attack_data[attack_data['groups'] == 0]['recovery_rate'].values[0]
                ax.axhline(y=baseline_rate, color=color['baseline'], linestyle='--', alpha=0.5)
                ax.text(max(groups), baseline_rate, f'{baseline_rate:.1f}% ',
                        va='bottom', ha='right', color=color['baseline'], fontsize=9)

        ax.set_xlabel('Number of adversarial interventions', fontsize=11)
        ax.set_ylabel('Recovery Rate (%)', fontsize=11)
        ax.set_title(attack_type.capitalize(), fontsize=12, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(all_groups)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
