import pandas as pd
import argparse
from load_data import WatermarkDataLoader
from typing import Dict, Any


def analyze_properly_encoded_tokens(min_tokens: int = 208):
    """
    Analyze properly encoded tokens across experiments that generated at least the specified number of tokens.
    Calculates encoding success rates and statistics per output folder.
    
    Args:
        min_tokens: Minimum number of tokens to filter for (default: 208)
    """
    # Initialize the data loader
    loader = WatermarkDataLoader()
    
    print("Available datasets:")
    available = loader.list_available_datasets()
    print(available.to_string(index=False))
    print()
    
    if len(available) == 0:
        print("No datasets found. Make sure you have run experiments and have data in the output/ directory.")
        return None, None, None
    
    # Load all datasets and analyze per folder
    all_data = []
    per_folder_analysis = {}
    
    print(f"\n{'='*80}")
    print(f"LOADING AND ANALYZING EACH OUTPUT FOLDER")
    print(f"{'='*80}")
    
    for dataset_info in loader.available_datasets:
        try:
            print(f"\nProcessing folder: {dataset_info['directory']}")
            print(f"  Model: {dataset_info['model']}")
            print(f"  Dataset: {dataset_info['dataset']}")
            print(f"  N-value: {dataset_info['n_value']}")
            print(f"  Timestamp: {dataset_info['timestamp']}")
            
            df = loader.load_dataset(dataset_info['directory'])
            
            # Check if required columns exist
            required_columns = ['properly_encoded_tokens', 'tokens_length']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"  WARNING: Missing required columns {missing_columns}. Skipping this folder.")
                continue
            
            # Add metadata columns
            df['dataset_name'] = dataset_info['dataset']
            df['model_name'] = dataset_info['model']
            df['n_value'] = dataset_info['n_value']
            df['experiment_timestamp'] = dataset_info['timestamp']
            df['output_folder'] = dataset_info['directory']
            
            # Analyze this specific folder for minimum tokens
            folder_target_df = df[df['tokens_length'] >= min_tokens].copy()
            
            # Always create folder analysis entry, even if no target-token experiments
            if len(folder_target_df) > 0:
                total_properly_encoded = folder_target_df['properly_encoded_tokens'].sum()
                total_tokens = folder_target_df['tokens_length'].sum()
                properly_encoded_ratio = total_properly_encoded / total_tokens if total_tokens > 0 else 0
                
                folder_analysis = {
                    'directory': dataset_info['directory'],
                    'model': dataset_info['model'],
                    'dataset': dataset_info['dataset'],
                    'n_value': dataset_info['n_value'],
                    'timestamp': dataset_info['timestamp'],
                    'total_experiments': len(df),
                    'experiments_target_tokens': len(folder_target_df),
                    'total_properly_encoded_target': total_properly_encoded,
                    'total_tokens_target': total_tokens,
                    'properly_encoded_ratio_target': properly_encoded_ratio,
                    'avg_properly_encoded_per_experiment': folder_target_df['properly_encoded_tokens'].mean(),
                    'avg_tokens_per_experiment': folder_target_df['tokens_length'].mean(),
                    'min_properly_encoded': folder_target_df['properly_encoded_tokens'].min(),
                    'max_properly_encoded': folder_target_df['properly_encoded_tokens'].max(),
                    'std_properly_encoded': folder_target_df['properly_encoded_tokens'].std()
                }
                
                print(f"  >={min_tokens}-token experiments: {len(folder_target_df)}/{len(df)}")
                print(f"  Total properly encoded tokens: {total_properly_encoded:,}/{total_tokens:,} ({properly_encoded_ratio:.2%})")
                print(f"  Avg properly encoded per experiment: {folder_analysis['avg_properly_encoded_per_experiment']:.1f}")
            else:
                # Create entry for folders with no minimum-token experiments
                folder_analysis = {
                    'directory': dataset_info['directory'],
                    'model': dataset_info['model'],
                    'dataset': dataset_info['dataset'],
                    'n_value': dataset_info['n_value'],
                    'timestamp': dataset_info['timestamp'],
                    'total_experiments': len(df),
                    'experiments_target_tokens': 0,
                    'total_properly_encoded_target': 0,
                    'total_tokens_target': 0,
                    'properly_encoded_ratio_target': 0.0,
                    'avg_properly_encoded_per_experiment': 0.0,
                    'avg_tokens_per_experiment': 0.0,
                    'min_properly_encoded': 0,
                    'max_properly_encoded': 0,
                    'std_properly_encoded': 0.0
                }
                print(f"  >={min_tokens}-token experiments: 0/{len(df)}")
                print(f"  No >={min_tokens}-token experiments found in this folder")
            
            per_folder_analysis[dataset_info['directory']] = folder_analysis
            
            all_data.append(df)
            
        except Exception as e:
            print(f"Error loading {dataset_info['directory']}: {e}")
            continue
    
    if not all_data:
        print("No data could be loaded.")
        return None, None, None
    
    # Combine all datasets
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\n{'='*80}")
    print(f"COMBINED ANALYSIS")
    print(f"{'='*80}")
    print(f"Total experiments across all folders: {len(combined_df)}")
    
    # Filter for minimum tokens
    filtered_df = combined_df[combined_df['tokens_length'] >= min_tokens].copy()
    
    print(f"Total experiments with >= {min_tokens} tokens: {len(filtered_df)}")
    
    if len(filtered_df) == 0:
        print(f"No experiments found with >= {min_tokens} tokens.")
        print(f"Available token lengths: {sorted(combined_df['tokens_length'].unique())}")
        return None, None, None
    
    # Check if properly_encoded_tokens column exists
    if 'properly_encoded_tokens' not in filtered_df.columns:
        print("ERROR: 'properly_encoded_tokens' column not found in data.")
        print("This suggests the data was generated before the properly_encoded_tokens feature was added.")
        print(f"Available columns: {list(filtered_df.columns)}")
        return None, None, None
    
    # Analyze properly encoded tokens
    total_properly_encoded = filtered_df['properly_encoded_tokens'].sum()
    total_tokens_generated = filtered_df['tokens_length'].sum()
    total_experiments = len(filtered_df)
    overall_properly_encoded_ratio = total_properly_encoded / total_tokens_generated if total_tokens_generated > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"PROPERLY ENCODED TOKENS ANALYSIS FOR >={min_tokens}-TOKEN EXPERIMENTS")
    print(f"{'='*60}")
    print(f"Total experiments with >= {min_tokens} tokens: {total_experiments:,}")
    print(f"Total tokens generated: {total_tokens_generated:,}")
    print(f"Total properly encoded tokens: {total_properly_encoded:,}")
    print(f"Overall properly encoded ratio: {overall_properly_encoded_ratio:.2%}")
    print(f"Average properly encoded tokens per experiment: {filtered_df['properly_encoded_tokens'].mean():.1f}")
    print(f"Average tokens per experiment: {filtered_df['tokens_length'].mean():.1f}")
    
    # Breakdown by model and dataset (including all combinations)
    print(f"\n{'='*60}")
    print(f"BREAKDOWN BY MODEL AND DATASET")
    print(f"{'='*60}")
    
    # Create breakdown from per-folder analysis to include all combinations
    breakdown_data = []
    for folder_key, folder_data in per_folder_analysis.items():
        if folder_data['experiments_target_tokens'] > 0:  # Only include folders with target experiments
            breakdown_data.append({
                'model_name': folder_data['model'],
                'dataset_name': folder_data['dataset'],
                'n_value': folder_data['n_value'],
                'total_experiments': folder_data['experiments_target_tokens'],
                'total_properly_encoded': folder_data['total_properly_encoded_target'],
                'total_tokens': folder_data['total_tokens_target'],
                'properly_encoded_ratio': folder_data['properly_encoded_ratio_target'],
                'avg_properly_encoded_per_exp': folder_data['avg_properly_encoded_per_experiment'],
                'avg_tokens_per_exp': folder_data['avg_tokens_per_experiment']
            })
    
    breakdown = pd.DataFrame(breakdown_data)
    breakdown = breakdown.sort_values(['model_name', 'dataset_name', 'n_value'])
    
    # Format the breakdown table
    print(f"{'Model':<35} {'Dataset':<45} {'N':<3} {f'>={min_tokens}-Exp':<7} {'Prop-Enc':<9} {'Total-Tok':<9} {'Ratio':<7} {'Avg-Prop':<8} {'Avg-Tok':<8}")
    print("-" * 140)
    
    for _, row in breakdown.iterrows():
        ratio_pct = f"{row['properly_encoded_ratio']:.1%}" if row['total_tokens'] > 0 else "N/A"
        prop_enc = f"{row['total_properly_encoded']:,}" if row['total_properly_encoded'] > 0 else "0"
        total_tok = f"{row['total_tokens']:,}"
        avg_prop = f"{row['avg_properly_encoded_per_exp']:.1f}"
        avg_tok = f"{row['avg_tokens_per_exp']:.1f}"
        
        print(f"{row['model_name']:<35} {row['dataset_name']:<45} {row['n_value']:<3} {row['total_experiments']:<7} {prop_enc:<9} {total_tok:<9} {ratio_pct:<7} {avg_prop:<8} {avg_tok:<8}")
    
    # Statistical analysis
    print(f"\n{'='*60}")
    print(f"STATISTICAL ANALYSIS (>={min_tokens}-TOKEN EXPERIMENTS)")
    print(f"{'='*60}")
    
    # Properly encoded tokens statistics
    prop_enc_stats = {
        'mean': filtered_df['properly_encoded_tokens'].mean(),
        'std': filtered_df['properly_encoded_tokens'].std(),
        'min': filtered_df['properly_encoded_tokens'].min(),
        'max': filtered_df['properly_encoded_tokens'].max(),
        'median': filtered_df['properly_encoded_tokens'].median(),
        'q25': filtered_df['properly_encoded_tokens'].quantile(0.25),
        'q75': filtered_df['properly_encoded_tokens'].quantile(0.75)
    }
    
    print(f"Properly Encoded Tokens per Experiment:")
    print(f"  Mean:   {prop_enc_stats['mean']:.2f}")
    print(f"  Std:    {prop_enc_stats['std']:.2f}")
    print(f"  Min:    {prop_enc_stats['min']}")
    print(f"  Max:    {prop_enc_stats['max']}")
    print(f"  Median: {prop_enc_stats['median']:.1f}")
    print(f"  Q25:    {prop_enc_stats['q25']:.1f}")
    print(f"  Q75:    {prop_enc_stats['q75']:.1f}")
    
    # Calculate per-experiment properly encoded ratios
    filtered_df['per_exp_properly_encoded_ratio'] = filtered_df['properly_encoded_tokens'] / filtered_df['tokens_length']
    
    ratio_stats = {
        'mean': filtered_df['per_exp_properly_encoded_ratio'].mean(),
        'std': filtered_df['per_exp_properly_encoded_ratio'].std(),
        'min': filtered_df['per_exp_properly_encoded_ratio'].min(),
        'max': filtered_df['per_exp_properly_encoded_ratio'].max(),
        'median': filtered_df['per_exp_properly_encoded_ratio'].median(),
        'q25': filtered_df['per_exp_properly_encoded_ratio'].quantile(0.25),
        'q75': filtered_df['per_exp_properly_encoded_ratio'].quantile(0.75)
    }
    
    print(f"\nProperly Encoded Ratio per Experiment:")
    print(f"  Mean:   {ratio_stats['mean']:.2%}")
    print(f"  Std:    {ratio_stats['std']:.2%}")
    print(f"  Min:    {ratio_stats['min']:.2%}")
    print(f"  Max:    {ratio_stats['max']:.2%}")
    print(f"  Median: {ratio_stats['median']:.2%}")
    print(f"  Q25:    {ratio_stats['q25']:.2%}")
    print(f"  Q75:    {ratio_stats['q75']:.2%}")
    
    # Success rate by field size (n value)
    if 'n_value' in filtered_df.columns:
        print(f"\nProperly encoded ratio by n value:")
        n_analysis = filtered_df.groupby('n_value').agg({
            'properly_encoded_tokens': 'sum',
            'tokens_length': 'sum'
        })
        n_analysis['properly_encoded_ratio'] = n_analysis['properly_encoded_tokens'] / n_analysis['tokens_length']
        n_analysis['properly_encoded_ratio'] = n_analysis['properly_encoded_ratio'].apply(lambda x: f"{x:.2%}")
        n_analysis['properly_encoded_tokens'] = n_analysis['properly_encoded_tokens'].apply(lambda x: f"{x:,}")
        n_analysis['tokens_length'] = n_analysis['tokens_length'].apply(lambda x: f"{x:,}")
        n_analysis.columns = ['properly_encoded_tokens', 'total_tokens', 'properly_encoded_ratio']
        print(n_analysis.to_string())
    
    # Per-folder analysis summary
    print(f"\n{'='*80}")
    print(f"PER-FOLDER ANALYSIS SUMMARY (>={min_tokens}-TOKEN EXPERIMENTS)")
    print(f"{'='*80}")
    
    if per_folder_analysis:
        # Create DataFrame for per-folder analysis
        folder_df = pd.DataFrame.from_dict(per_folder_analysis, orient='index')
        folder_df = folder_df.sort_values(['model', 'dataset', 'n_value'])
        
        print(f"\nDetailed per-folder results:")
        for _, row in folder_df.iterrows():
            print(f"\nFolder: {row['directory']}")
            print(f"  Model: {row['model']}")
            print(f"  Dataset: {row['dataset']}")
            print(f"  N-value: {row['n_value']}")
            print(f"  >={min_tokens}-token experiments: {row['experiments_target_tokens']}/{row['total_experiments']}")
            
            if row['experiments_target_tokens'] > 0:
                print(f"  Total tokens: {row['total_tokens_target']:,}")
                print(f"  Properly encoded tokens: {row['total_properly_encoded_target']:,}")
                print(f"  Properly encoded ratio: {row['properly_encoded_ratio_target']:.2%}")
                print(f"  Avg properly encoded per experiment: {row['avg_properly_encoded_per_experiment']:.1f}")
                print(f"  Avg tokens per experiment: {row['avg_tokens_per_experiment']:.1f}")
                print(f"  Statistics - Min: {row['min_properly_encoded']}, Max: {row['max_properly_encoded']}, Std: {row['std_properly_encoded']:.1f}")
            else:
                print(f"  No experiments with >={min_tokens} tokens")
        
        print(f"\nTotal folders analyzed: {len(folder_df)}")
    
    return filtered_df, prop_enc_stats, per_folder_analysis


def main():
    """
    Main function to run the properly encoded tokens analysis with command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Analyze properly encoded tokens by minimum token count")
    parser.add_argument("--min-tokens", type=int, default=208, 
                        help="Minimum number of tokens to filter for (default: 208)")
    
    args = parser.parse_args()
    
    print(f"Analyzing properly encoded tokens for experiments with >= {args.min_tokens} tokens...")
    
    try:
        filtered_data, stats, breakdown = analyze_properly_encoded_tokens(args.min_tokens)
        return filtered_data, stats, breakdown
    except Exception as e:
        print(f"Error during analysis: {e}")
        return None, None, None


if __name__ == "__main__":
    filtered_data, stats, breakdown = main()
