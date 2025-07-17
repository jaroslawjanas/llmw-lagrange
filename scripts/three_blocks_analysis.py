import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
from datetime import datetime
from load_data import WatermarkDataLoader
from typing import Dict, List, Tuple


def analyze_threshold_data(df: pd.DataFrame, step: int, max_tokens: int) -> Tuple[List[int], List[int], List[int]]:
    """
    Analyze data for different token thresholds.
    
    Args:
        df: DataFrame with experiment data
        step: Step size for thresholds
        max_tokens: Maximum token threshold
        
    Returns:
        Tuple of (thresholds, total_counts, success_counts)
    """
    thresholds = list(range(0, max_tokens + 1, step))
    total_counts = []
    success_counts = []
    
    for threshold in thresholds:
        # Filter data for this threshold
        filtered_data = df[df['tokens_length'] >= threshold]
        total_count = len(filtered_data)
        
        # Count generations with at least 3 matching blocks
        success_count = len(filtered_data[filtered_data['matching_blocks'] >= 3])
        
        total_counts.append(total_count)
        success_counts.append(success_count)
    
    return thresholds, total_counts, success_counts


def create_histogram(thresholds: List[int], total_counts: List[int], success_counts: List[int], 
                    experiment_info: Dict, output_path: str) -> None:
    """
    Create and save a histogram for the three blocks analysis.
    
    Args:
        thresholds: List of threshold values
        total_counts: List of total generation counts for each threshold
        success_counts: List of successful generation counts for each threshold
        experiment_info: Dictionary with experiment metadata
        output_path: Path to save the histogram
    """
    plt.figure(figsize=(12, 8))
    
    # Create bar width
    bar_width = (thresholds[1] - thresholds[0]) * 0.8 if len(thresholds) > 1 else 1
    
    # Create overlaid bars
    plt.bar(thresholds, total_counts, width=bar_width, alpha=0.4, color='lightgray', 
            label='Total Generations', edgecolor='gray', linewidth=0.5)
    plt.bar(thresholds, success_counts, width=bar_width, alpha=0.8, color='steelblue', 
            label='Generations with ≥3 Matching Blocks', edgecolor='darkblue', linewidth=0.5)
    
    # Customize the plot
    plt.xlabel('Minimum Token Length Threshold', fontsize=12)
    plt.ylabel('Number of Generations', fontsize=12)
    
    # Create title with experiment info
    title = f"Three Blocks Analysis: Success vs Token Length Threshold\n"
    title += f"Model: {experiment_info['model']} | Dataset: {experiment_info['dataset']} | N-value: {experiment_info['n_value']}"
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add legend
    plt.legend(fontsize=11)
    
    # Set axis limits and ticks
    plt.xlim(-bar_width/2, max(thresholds) + bar_width/2)
    plt.ylim(0, max(max(total_counts), max(success_counts)) * 1.1)
    
    # Improve tick spacing
    if len(thresholds) > 20:
        # Show every 5th threshold for readability
        tick_indices = range(0, len(thresholds), 5)
        plt.xticks([thresholds[i] for i in tick_indices])
    
    # Add text annotation with success rate at zero threshold
    if total_counts[0] > 0:
        success_rate = success_counts[0] / total_counts[0] * 100
        plt.text(0.02, 0.98, f'Overall Success Rate: {success_rate:.1f}%', 
                transform=plt.gca().transAxes, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                verticalalignment='top')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_summary_statistics(all_results: Dict, step: int, max_tokens: int, output_dir: str) -> None:
    """
    Generate a summary CSV with three blocks analysis statistics.
    
    Args:
        all_results: Dictionary with results for all experiments
        step: Step size used for thresholds
        max_tokens: Maximum token threshold used
        output_dir: Output directory for the summary file
    """
    summary_data = []
    
    for experiment_name, (thresholds, total_counts, success_counts, experiment_info) in all_results.items():
        for i, threshold in enumerate(thresholds):
            total = total_counts[i]
            success = success_counts[i]
            success_rate = (success / total * 100) if total > 0 else 0
            
            summary_data.append({
                'experiment': experiment_name,
                'model': experiment_info['model'],
                'dataset': experiment_info['dataset'],
                'n_value': experiment_info['n_value'],
                'timestamp': experiment_info['timestamp'],
                'threshold': threshold,
                'total_generations': total,
                'successful_generations': success,
                'success_rate_percent': success_rate
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(output_dir, 'three_blocks_analysis_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    
    print(f"Summary statistics saved to: {summary_file}")


def main():
    """
    Main function to run three blocks analysis with command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Analyze generations with 3+ matching blocks by token length thresholds")
    parser.add_argument("--step", type=int, required=True,
                        help="Step size for token thresholds (required)")
    parser.add_argument("--max-tokens", type=int, required=True,
                        help="Maximum token threshold (required)")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Directory to save histogram images (default: output)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.step <= 0:
        print("ERROR: --step must be a positive integer")
        return 1
    
    if args.max_tokens <= 0:
        print("ERROR: --max-tokens must be a positive integer")
        return 1
    
    if args.max_tokens < args.step:
        print("ERROR: --max-tokens must be greater than or equal to --step")
        return 1
    
    print(f"Three Blocks Analysis Configuration:")
    print(f"  Step size: {args.step}")
    print(f"  Maximum tokens: {args.max_tokens}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Thresholds: {list(range(0, args.max_tokens + 1, args.step))}")
    print()
    
    # Initialize the data loader
    loader = WatermarkDataLoader()
    
    print("Available datasets:")
    available = loader.list_available_datasets()
    print(available.to_string(index=False))
    print()
    
    if len(available) == 0:
        print("No datasets found. Make sure you have run experiments and have data in the output/ directory.")
        return 1
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_subdir = f"three_blocks_histograms_{timestamp}"
    full_output_dir = os.path.join(args.output_dir, output_subdir)
    os.makedirs(full_output_dir, exist_ok=True)
    
    print(f"Creating histograms in: {full_output_dir}")
    print()
    
    # Store results for summary
    all_results = {}
    
    # Process each experiment folder
    for dataset_info in loader.available_datasets:
        try:
            print(f"Processing: {dataset_info['directory']}")
            print(f"  Model: {dataset_info['model']}")
            print(f"  Dataset: {dataset_info['dataset']}")
            print(f"  N-value: {dataset_info['n_value']}")
            print(f"  Timestamp: {dataset_info['timestamp']}")
            
            # Load the dataset
            df = loader.load_dataset(dataset_info['directory'])
            
            # Check if required columns exist
            if 'tokens_length' not in df.columns or 'matching_blocks' not in df.columns:
                print(f"  WARNING: Missing required columns (tokens_length, matching_blocks). Skipping.")
                continue
            
            # Perform threshold analysis
            thresholds, total_counts, success_counts = analyze_threshold_data(df, args.step, args.max_tokens)
            
            # Create histogram
            output_filename = f"{dataset_info['directory']}.png"
            output_path = os.path.join(full_output_dir, output_filename)
            
            create_histogram(thresholds, total_counts, success_counts, dataset_info, output_path)
            
            # Store results for summary
            all_results[dataset_info['directory']] = (thresholds, total_counts, success_counts, dataset_info)
            
            # Print some statistics
            total_experiments = len(df)
            if total_experiments > 0:
                overall_success_rate = len(df[df['matching_blocks'] >= 3]) / total_experiments * 100
                print(f"  Total experiments: {total_experiments}")
                print(f"  Overall success rate (≥3 blocks): {overall_success_rate:.1f}%")
                print(f"  Histogram saved: {output_filename}")
            
            print()
            
        except Exception as e:
            print(f"  ERROR processing {dataset_info['directory']}: {e}")
            print()
            continue
    
    if all_results:
        # Generate summary statistics
        generate_summary_statistics(all_results, args.step, args.max_tokens, full_output_dir)
        
        print(f"\n{'='*60}")
        print(f"THREE BLOCKS ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"Processed {len(all_results)} experiment folders")
        print(f"Histograms saved in: {full_output_dir}")
        print(f"Step size: {args.step}")
        print(f"Max tokens: {args.max_tokens}")
        print(f"Total thresholds analyzed: {len(range(0, args.max_tokens + 1, args.step))}")
        
        # Show some overall statistics
        print(f"\nExperiment Summary:")
        for experiment_name, (thresholds, total_counts, success_counts, experiment_info) in all_results.items():
            if total_counts[0] > 0:  # At threshold 0
                success_rate = success_counts[0] / total_counts[0] * 100
                print(f"  {experiment_info['model']} | {experiment_info['dataset']} | n{experiment_info['n_value']}: {success_rate:.1f}% success rate (≥3 blocks)")
    
    else:
        print("No valid datasets were processed.")
        return 1


if __name__ == "__main__":
    main()
