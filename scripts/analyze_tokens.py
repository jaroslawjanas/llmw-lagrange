import pandas as pd
import argparse
from load_data import WatermarkDataLoader
from typing import Dict, Any


def analyze_token_experiments(min_tokens: int = 704):
    """
    Analyze experiments that generated at least the specified number of tokens.
    Calculates watermark success rates and timing statistics per output folder.
    
    Args:
        min_tokens: Minimum number of tokens to filter for (default: 704)
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
                folder_analysis = {
                    'directory': dataset_info['directory'],
                    'model': dataset_info['model'],
                    'dataset': dataset_info['dataset'],
                    'n_value': dataset_info['n_value'],
                    'timestamp': dataset_info['timestamp'],
                    'total_experiments': len(df),
                    'experiments_target_tokens': len(folder_target_df),
                    'positive_watermarks_target': folder_target_df['watermark_recovered'].sum(),
                    'success_rate_target': folder_target_df['watermark_recovered'].mean(),
                    'avg_matching_blocks_target': folder_target_df['matching_blocks'].mean(),
                    'avg_encoding_time_target': folder_target_df['encoding_time'].mean(),
                    'avg_decoding_time_target': folder_target_df['decoding_time'].mean(),
                    'avg_mcp_time_target': folder_target_df['mcp_time'].mean(),
                    'avg_total_time_target': (folder_target_df['encoding_time'] + 
                                            folder_target_df['decoding_time'] + 
                                            folder_target_df['mcp_time']).mean()
                }
                
                print(f"  >={min_tokens}-token experiments: {len(folder_target_df)}/{len(df)}")
                print(f"  Success rate: {folder_analysis['success_rate_target']:.2%}")
                print(f"  Avg matching blocks: {folder_analysis['avg_matching_blocks_target']:.2f}")
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
                    'positive_watermarks_target': 0,
                    'success_rate_target': 0.0,
                    'avg_matching_blocks_target': 0.0,
                    'avg_encoding_time_target': 0.0,
                    'avg_decoding_time_target': 0.0,
                    'avg_mcp_time_target': 0.0,
                    'avg_total_time_target': 0.0
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
    
    # Analyze watermark success
    positive_watermarks = filtered_df['watermark_recovered'].sum()
    total_experiments = len(filtered_df)
    success_rate = positive_watermarks / total_experiments
    
    print(f"\n{'='*60}")
    print(f"WATERMARK ANALYSIS FOR >={min_tokens}-TOKEN EXPERIMENTS")
    print(f"{'='*60}")
    print(f"Total experiments with >= {min_tokens} tokens: {total_experiments}")
    print(f"Positive watermarks detected: {positive_watermarks}")
    print(f"Watermark success rate: {success_rate:.2%}")
    
    # Breakdown by model and dataset (including all combinations)
    print(f"\n{'='*60}")
    print(f"BREAKDOWN BY MODEL AND DATASET")
    print(f"{'='*60}")
    
    # Create breakdown from per-folder analysis to include all combinations
    breakdown_data = []
    for folder_key, folder_data in per_folder_analysis.items():
        breakdown_data.append({
            'model_name': folder_data['model'],
            'dataset_name': folder_data['dataset'],
            'n_value': folder_data['n_value'],
            'total_experiments': folder_data['experiments_target_tokens'],
            'positive_watermarks': folder_data['positive_watermarks_target'],
            'success_rate': folder_data['success_rate_target'],
            'avg_matching_blocks': folder_data['avg_matching_blocks_target'],
            'avg_encoding_time': folder_data['avg_encoding_time_target'],
            'avg_decoding_time': folder_data['avg_decoding_time_target'],
            'avg_mcp_time': folder_data['avg_mcp_time_target']
        })
    
    breakdown = pd.DataFrame(breakdown_data)
    breakdown = breakdown.sort_values(['model_name', 'dataset_name', 'n_value'])
    
    # Format the breakdown table
    print(f"{'Model':<35} {'Dataset':<45} {'N':<3} {f'>={min_tokens}-Exp':<7} {'Pos-WM':<6} {'Success':<8} {'Blocks':<6} {'Enc-Time':<8} {'Dec-Time':<8} {'MCP-Time':<8}")
    print("-" * 140)
    
    for _, row in breakdown.iterrows():
        success_pct = f"{row['success_rate']:.1%}" if row['total_experiments'] > 0 else "N/A"
        blocks = f"{row['avg_matching_blocks']:.1f}" if row['total_experiments'] > 0 else "N/A"
        enc_time = f"{row['avg_encoding_time']:.2f}s" if row['total_experiments'] > 0 else "N/A"
        dec_time = f"{row['avg_decoding_time']:.3f}s" if row['total_experiments'] > 0 else "N/A"
        mcp_time = f"{row['avg_mcp_time']:.3f}s" if row['total_experiments'] > 0 else "N/A"
        
        print(f"{row['model_name']:<35} {row['dataset_name']:<45} {row['n_value']:<3} {row['total_experiments']:<7} {row['positive_watermarks']:<6} {success_pct:<8} {blocks:<6} {enc_time:<8} {dec_time:<8} {mcp_time:<8}")
    
    # Merged across datasets by model and n (weighted by number of filtered generations)
    print(f"\n{'='*60}")
    print(f"MERGED ACROSS DATASETS BY MODEL AND N (WEIGHTED)")
    print(f"{'='*60}")

    # Aggregate over filtered_df, which already enforces min_tokens
    timing_cols = ['encoding_time', 'decoding_time', 'mcp_time']
    agg_spec = {
        'total_experiments': ('watermark_recovered', 'size'),
        'positive_watermarks': ('watermark_recovered', 'sum'),
        'success_rate': ('watermark_recovered', 'mean'),
    }
    if 'matching_blocks' in filtered_df.columns:
        agg_spec['avg_matching_blocks'] = ('matching_blocks', 'mean')
    # Add timing column means if present
    for col in timing_cols:
        if col in filtered_df.columns:
            agg_spec[f'avg_{col}'] = (col, 'mean')

    merged = (
        filtered_df
        .groupby(['model_name', 'n_value'])
        .agg(**agg_spec)
        .reset_index()
        .sort_values(['model_name', 'n_value'])
    )

    # Header
    print(f"{'Model':<35} {'N':<3} {f'>={min_tokens}-Exp':<7} {'Pos-WM':<6} {'Success':<8} {'Blocks':<6} {'Enc-Time':<8} {'Dec-Time':<8} {'MCP-Time':<8}")
    print("-" * 120)

    has_blocks = 'avg_matching_blocks' in merged.columns
    has_enc = 'avg_encoding_time' in merged.columns
    has_dec = 'avg_decoding_time' in merged.columns
    has_mcp = 'avg_mcp_time' in merged.columns

    for _, row in merged.iterrows():
        total = int(row['total_experiments'])
        pos = int(row['positive_watermarks'])
        success_pct = f"{(pos / total):.1%}" if total > 0 else "N/A"
        blocks = f"{row['avg_matching_blocks']:.1f}" if has_blocks else "N/A"
        enc_time = f"{row['avg_encoding_time']:.2f}s" if has_enc else "N/A"
        dec_time = f"{row['avg_decoding_time']:.3f}s" if has_dec else "N/A"
        mcp_time = f"{row['avg_mcp_time']:.3f}s" if has_mcp else "N/A"

        print(f"{row['model_name']:<35} {row['n_value']:<3} {total:<7} {pos:<6} {success_pct:<8} {blocks:<6} {enc_time:<8} {dec_time:<8} {mcp_time:<8}")

    # Aggregated across models by n (weighted by number of filtered generations)
    print(f"\n{'='*60}")
    print(f"AGGREGATED ACROSS MODELS BY N (WEIGHTED)")
    print(f"{'='*60}")
    
    timing_cols_models = ['encoding_time', 'decoding_time', 'mcp_time']
    agg_spec_models = {
        'total_experiments': ('watermark_recovered', 'size'),
        'positive_watermarks': ('watermark_recovered', 'sum'),
        'success_rate': ('watermark_recovered', 'mean'),
    }
    if 'matching_blocks' in filtered_df.columns:
        agg_spec_models['avg_matching_blocks'] = ('matching_blocks', 'mean')
    for col in timing_cols_models:
        if col in filtered_df.columns:
            agg_spec_models[f'avg_{col}'] = (col, 'mean')
    
    by_n = (
        filtered_df
        .groupby(['n_value'])
        .agg(**agg_spec_models)
        .reset_index()
        .sort_values(['n_value'])
    )
    
    # Header
    print(f"{'N':<3} {f'>={min_tokens}-Exp':<7} {'Pos-WM':<6} {'Success':<8} {'Blocks':<6} {'Enc-Time':<8} {'Dec-Time':<8} {'MCP-Time':<8}")
    print("-" * 90)
    
    has_blocks_n = 'avg_matching_blocks' in by_n.columns
    has_enc_n = 'avg_encoding_time' in by_n.columns
    has_dec_n = 'avg_decoding_time' in by_n.columns
    has_mcp_n = 'avg_mcp_time' in by_n.columns
    
    for _, row in by_n.iterrows():
        total = int(row['total_experiments'])
        pos = int(row['positive_watermarks'])
        success_pct = f"{(pos / total):.1%}" if total > 0 else "N/A"
        blocks = f"{row['avg_matching_blocks']:.1f}" if has_blocks_n else "N/A"
        enc_time = f"{row['avg_encoding_time']:.2f}s" if has_enc_n else "N/A"
        dec_time = f"{row['avg_decoding_time']:.3f}s" if has_dec_n else "N/A"
        mcp_time = f"{row['avg_mcp_time']:.3f}s" if has_mcp_n else "N/A"
    
        print(f"{row['n_value']:<3} {total:<7} {pos:<6} {success_pct:<8} {blocks:<6} {enc_time:<8} {dec_time:<8} {mcp_time:<8}")
    
    # Calculate overall timing statistics
    print(f"\n{'='*60}")
    print(f"TIMING STATISTICS (>={min_tokens}-TOKEN EXPERIMENTS)")
    print(f"{'='*60}")
    
    timing_stats = {}
    timing_columns = ['encoding_time', 'decoding_time', 'mcp_time']
    
    for col in timing_columns:
        if col in filtered_df.columns:
            stats = {
                'mean': filtered_df[col].mean(),
                'std': filtered_df[col].std(),
                'min': filtered_df[col].min(),
                'max': filtered_df[col].max(),
                'median': filtered_df[col].median()
            }
            timing_stats[col] = stats
            
            print(f"\n{col.replace('_', ' ').title()}:")
            print(f"  Mean: {stats['mean']:.4f}s")
            print(f"  Std:  {stats['std']:.4f}s")
            print(f"  Min:  {stats['min']:.4f}s")
            print(f"  Max:  {stats['max']:.4f}s")
            print(f"  Median: {stats['median']:.4f}s")
    
    # Total time analysis
    if all(col in filtered_df.columns for col in timing_columns):
        filtered_df['total_time'] = filtered_df[timing_columns].sum(axis=1)
        total_time_stats = {
            'mean': filtered_df['total_time'].mean(),
            'std': filtered_df['total_time'].std(),
            'min': filtered_df['total_time'].min(),
            'max': filtered_df['total_time'].max(),
            'median': filtered_df['total_time'].median()
        }
        
        print(f"\nTotal Time (Encoding + Decoding + MCP):")
        print(f"  Mean: {total_time_stats['mean']:.4f}s")
        print(f"  Std:  {total_time_stats['std']:.4f}s")
        print(f"  Min:  {total_time_stats['min']:.4f}s")
        print(f"  Max:  {total_time_stats['max']:.4f}s")
        print(f"  Median: {total_time_stats['median']:.4f}s")
    
    # Additional analysis
    print(f"\n{'='*60}")
    print(f"ADDITIONAL STATISTICS")
    print(f"{'='*60}")
    
    if 'matching_blocks' in filtered_df.columns:
        avg_matching_blocks = filtered_df['matching_blocks'].mean()
        print(f"Average matching blocks: {avg_matching_blocks:.2f}")
    
    if 'field_size' in filtered_df.columns:
        field_sizes = filtered_df['field_size'].unique()
        print(f"Field sizes used: {sorted(field_sizes)}")
    
    # Success rate by field size (n value)
    if 'n_value' in filtered_df.columns:
        print(f"\nSuccess rate by n value:")
        n_analysis = filtered_df.groupby('n_value')['watermark_recovered'].agg(['count', 'sum', 'mean'])
        n_analysis.columns = ['total_experiments', 'positive_watermarks', 'success_rate']
        n_analysis['success_rate'] = n_analysis['success_rate'].apply(lambda x: f"{x:.2%}")
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
            print(f"  Success rate: {row['success_rate_target']:.2%} ({row['positive_watermarks_target']}/{row['experiments_target_tokens']})")
            print(f"  Avg matching blocks: {row['avg_matching_blocks_target']:.2f}")
            print(f"  Timing - Encoding: {row['avg_encoding_time_target']:.3f}s, Decoding: {row['avg_decoding_time_target']:.3f}s, MCP: {row['avg_mcp_time_target']:.3f}s")
            print(f"  Total time: {row['avg_total_time_target']:.3f}s")
        
        print(f"\nTotal folders analyzed: {len(folder_df)}")
    
    return filtered_df, timing_stats, per_folder_analysis


def main():
    """
    Main function to run the token analysis with command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Analyze watermarking experiments by minimum token count")
    parser.add_argument("--min-tokens", type=int, default=704, 
                        help="Minimum number of tokens to filter for (default: 704)")
    
    args = parser.parse_args()
    
    print(f"Analyzing experiments with >= {args.min_tokens} tokens...")
    
    try:
        filtered_data, timing_stats, breakdown = analyze_token_experiments(args.min_tokens)
        return filtered_data, timing_stats, breakdown
    except Exception as e:
        print(f"Error during analysis: {e}")
        return None, None, None


if __name__ == "__main__":
    filtered_data, timing_stats, breakdown = main()
