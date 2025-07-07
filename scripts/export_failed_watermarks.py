import pandas as pd
import argparse
from load_data import WatermarkDataLoader
from typing import Optional
import os


def export_failed_watermarks(min_tokens: int = 704, output_file: Optional[str] = None):
    """
    Export prompts for failed watermark experiments to CSV.
    Filters experiments by minimum token count and exports only those where watermark recovery failed.
    
    Args:
        min_tokens: Minimum number of tokens to filter for (default: 704)
        output_file: Output CSV filename (default: auto-generated based on min_tokens)
    """
    # Initialize the data loader
    loader = WatermarkDataLoader()
    
    print("Available datasets:")
    available = loader.list_available_datasets()
    print(available.to_string(index=False))
    print()
    
    if len(available) == 0:
        print("No datasets found. Make sure you have run experiments and have data in the output/ directory.")
        return None
    
    # Load all datasets and combine
    all_data = []
    
    print(f"\n{'='*80}")
    print(f"LOADING ALL DATASETS")
    print(f"{'='*80}")
    
    for dataset_info in loader.available_datasets:
        try:
            print(f"\nProcessing folder: {dataset_info['directory']}")
            print(f"  Model: {dataset_info['model']}")
            print(f"  Dataset: {dataset_info['dataset']}")
            print(f"  N-value: {dataset_info['n_value']}")
            print(f"  Timestamp: {dataset_info['timestamp']}")
            
            df = loader.load_dataset(dataset_info['directory'])
            
            # Add metadata columns for tracking
            df['dataset_name'] = dataset_info['dataset']
            df['model_name'] = dataset_info['model']
            df['n_value'] = dataset_info['n_value']
            df['experiment_timestamp'] = dataset_info['timestamp']
            df['output_folder'] = dataset_info['directory']
            
            all_data.append(df)
            print(f"  Loaded {len(df)} experiments")
            
        except Exception as e:
            print(f"Error loading {dataset_info['directory']}: {e}")
            continue
    
    if not all_data:
        print("No data could be loaded.")
        return None
    
    # Combine all datasets
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\n{'='*80}")
    print(f"FILTERING AND ANALYSIS")
    print(f"{'='*80}")
    print(f"Total experiments across all folders: {len(combined_df)}")
    
    # Filter for minimum tokens
    filtered_df = combined_df[combined_df['tokens_length'] >= min_tokens].copy()
    print(f"Experiments with >= {min_tokens} tokens: {len(filtered_df)}")
    
    if len(filtered_df) == 0:
        print(f"No experiments found with >= {min_tokens} tokens.")
        print(f"Available token lengths: {sorted(combined_df['tokens_length'].unique())}")
        return None
    
    # Filter for failed watermarks
    if 'watermark_recovered' not in filtered_df.columns:
        print("Error: 'watermark_recovered' column not found in data.")
        print(f"Available columns: {list(filtered_df.columns)}")
        return None
    
    failed_watermarks = filtered_df[filtered_df['watermark_recovered'] == False].copy()
    print(f"Failed watermark experiments (>= {min_tokens} tokens): {len(failed_watermarks)}")
    
    if len(failed_watermarks) == 0:
        print("No failed watermark experiments found in the filtered data.")
        success_rate = filtered_df['watermark_recovered'].mean()
        print(f"Success rate for >= {min_tokens} tokens: {success_rate:.2%}")
        return None
    
    # Prepare data for export
    print(f"\n{'='*80}")
    print(f"PREPARING EXPORT DATA")
    print(f"{'='*80}")
    
    # Select relevant columns for export
    export_columns = [
        'prompt',
        'generated_text', 
        'tokens_length',
        'watermark_recovered',
        'matching_blocks',
        'model_name',
        'dataset_name',
        'n_value',
        'output_folder',
        'experiment_timestamp'
    ]
    
    # Add optional columns if they exist
    optional_columns = [
        'field_size',
        'encoding_time',
        'decoding_time',
        'mcp_time',
        'watermark_blocks',
        'decoded_blocks'
    ]
    
    for col in optional_columns:
        if col in failed_watermarks.columns:
            export_columns.append(col)
    
    # Filter to only include columns that exist
    available_export_columns = [col for col in export_columns if col in failed_watermarks.columns]
    export_df = failed_watermarks[available_export_columns].copy()
    
    # Sort by model, dataset, and token length for better organization
    sort_columns = ['model_name', 'dataset_name', 'tokens_length']
    export_df = export_df.sort_values(sort_columns)
    
    print(f"Export columns: {available_export_columns}")
    print(f"Rows to export: {len(export_df)}")
    
    # Generate output filename if not provided
    if output_file is None:
        output_file = f"failed_watermarks_min{min_tokens}tokens.csv"
    
    # Ensure output file has .csv extension
    if not output_file.endswith('.csv'):
        output_file += '.csv'
    
    # Create scripts/output directory if it doesn't exist and prepend to output path
    scripts_output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(scripts_output_dir, exist_ok=True)
    output_file = os.path.join(scripts_output_dir, output_file)
    
    # Export to CSV
    try:
        export_df.to_csv(output_file, index=False)
        print(f"\n{'='*80}")
        print(f"EXPORT COMPLETE")
        print(f"{'='*80}")
        print(f"Successfully exported {len(export_df)} failed watermark experiments to: {output_file}")
        
        # Print summary statistics
        print(f"\nSummary by model and dataset:")
        summary = export_df.groupby(['model_name', 'dataset_name']).agg({
            'prompt': 'count',
            'tokens_length': ['mean', 'min', 'max'],
            'matching_blocks': 'mean'
        }).round(2)
        
        summary.columns = ['failed_count', 'avg_tokens', 'min_tokens', 'max_tokens', 'avg_matching_blocks']
        print(summary.to_string())
        
        # Print token length distribution
        print(f"\nToken length distribution of failed experiments:")
        token_dist = export_df['tokens_length'].value_counts().sort_index()
        print(token_dist.to_string())
        
        return export_df
        
    except Exception as e:
        print(f"Error writing to CSV file: {e}")
        return None


def main():
    """
    Main function to run the failed watermark export with command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Export prompts for failed watermark experiments to CSV")
    parser.add_argument("--min-tokens", type=int, default=704, 
                        help="Minimum number of tokens to filter for (default: 704)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV filename (default: auto-generated)")
    
    args = parser.parse_args()
    
    print(f"Exporting failed watermark experiments with >= {args.min_tokens} tokens...")
    
    try:
        result = export_failed_watermarks(args.min_tokens, args.output)
        
        if result is not None:
            print(f"\nExport completed successfully!")
            print(f"You can now analyze the failed experiments in the CSV file.")
        else:
            print(f"\nExport failed or no data to export.")
            
        return result
        
    except Exception as e:
        print(f"Error during export: {e}")
        return None


if __name__ == "__main__":
    result = main()
