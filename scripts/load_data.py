import pandas as pd
import json
import os
from typing import List, Dict, Optional, Union
from pathlib import Path


class WatermarkDataLoader:
    """
    Data loader for LLM watermarking experiment results.
    Loads data from parquet files in the output directory, preserving data types
    and providing utilities for working with the structured data.
    """
    
    def __init__(self, output_dir: str = "../output"):
        """
        Initialize the data loader.
        
        Args:
            output_dir: Path to the output directory containing experiment results
        """
        self.output_dir = Path(output_dir)
        self.available_datasets = self._discover_datasets()
    
    def _discover_datasets(self) -> List[Dict[str, str]]:
        """
        Discover available datasets in the output directory.
        
        Returns:
            List of dictionaries containing dataset information
        """
        datasets = []
        
        if not self.output_dir.exists():
            print(f"Warning: Output directory '{self.output_dir}' does not exist")
            return datasets
        
        for subdir in self.output_dir.iterdir():
            if subdir.is_dir():
                parquet_file = subdir / "statistics.parquet"
                csv_file = subdir / "statistics.csv"
                
                if parquet_file.exists() or csv_file.exists():
                    # Parse directory name to extract metadata
                    # Format: <company>_<modelname>_<dataset org>_<dataset name>_n<field size n>_<date>_<time>
                    parts = subdir.name.split('_')
                    
                    # Initialize defaults
                    company = "unknown"
                    model_name = "unknown"
                    dataset_org = "unknown"
                    dataset_name = "unknown"
                    n_value = "unknown"
                    date_part = "unknown"
                    time_part = "unknown"
                    
                    if len(parts) >= 7:
                        # Find the n-value part (starts with 'n' followed by digits)
                        n_index = None
                        for i, part in enumerate(parts):
                            if part.startswith('n') and part[1:].isdigit():
                                n_index = i
                                n_value = part[1:]  # Remove 'n' prefix
                                break
                        
                        if n_index is not None and n_index >= 4:
                            # Extract parts based on known positions
                            company = parts[0]
                            model_name = parts[1]
                            dataset_org = parts[2]
                            dataset_name = parts[3]
                            
                            # Handle cases where dataset name might have multiple parts
                            if n_index > 4:
                                dataset_name = '_'.join(parts[3:n_index])
                            
                            # Date and time are after n-value
                            if n_index + 1 < len(parts):
                                date_part = parts[n_index + 1]
                            if n_index + 2 < len(parts):
                                time_part = parts[n_index + 2]
                    
                    # Combine for display
                    model = f"{company}_{model_name}"
                    dataset = f"{dataset_org}_{dataset_name}"
                    timestamp = f"{date_part}_{time_part}"
                    
                    datasets.append({
                        'directory': subdir.name,
                        'model': model,
                        'dataset': dataset,
                        'n_value': n_value,
                        'timestamp': timestamp,
                        'has_parquet': parquet_file.exists(),
                        'has_csv': csv_file.exists(),
                        'path': str(subdir)
                    })
        
        return sorted(datasets, key=lambda x: x['timestamp'], reverse=True)
    
    def list_available_datasets(self) -> pd.DataFrame:
        """
        List all available datasets in a readable format.
        
        Returns:
            DataFrame with information about available datasets
        """
        if not self.available_datasets:
            print("No datasets found in the output directory.")
            return pd.DataFrame()
        
        df = pd.DataFrame(self.available_datasets)
        return df[['directory', 'model', 'dataset', 'n_value', 'timestamp', 'has_parquet', 'has_csv']]
    
    def load_dataset(self, directory_name: str, use_parquet: bool = True) -> pd.DataFrame:
        """
        Load a specific dataset by directory name.
        
        Args:
            directory_name: Name of the directory containing the dataset
            use_parquet: Whether to use parquet format (recommended) or CSV
            
        Returns:
            DataFrame with the loaded data
        """
        dataset_path = self.output_dir / directory_name
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset directory '{directory_name}' not found")
        
        parquet_file = dataset_path / "statistics.parquet"
        csv_file = dataset_path / "statistics.csv"
        
        # Try parquet first if requested and available
        if use_parquet and parquet_file.exists():
            try:
                print(f"Loading data from: {parquet_file}")
                df = pd.read_parquet(parquet_file)
                print(f"Successfully loaded {len(df)} rows with {len(df.columns)} columns from parquet")
                return df
            except Exception as e:
                print(f"Error reading parquet file: {e}")
                print(f"Falling back to CSV format...")
        
        # Fall back to CSV
        if csv_file.exists():
            print(f"Loading data from: {csv_file}")
            df = pd.read_csv(csv_file)
            print(f"Successfully loaded {len(df)} rows with {len(df.columns)} columns from CSV")
            return df
        
        # No files found
        raise FileNotFoundError(f"No data file (parquet or CSV) found in '{directory_name}'")
    
    def load_latest_dataset(self, model: Optional[str] = None, 
                          dataset: Optional[str] = None, 
                          n_value: Optional[Union[str, int]] = None) -> pd.DataFrame:
        """
        Load the most recent dataset, optionally filtered by criteria.
        
        Args:
            model: Filter by model name (e.g., "meta-llama")
            dataset: Filter by dataset name (e.g., "ChristophSchuhmann")
            n_value: Filter by n value (e.g., 8 or "8")
            
        Returns:
            DataFrame with the loaded data
        """
        filtered_datasets = self.available_datasets.copy()
        
        if model:
            filtered_datasets = [d for d in filtered_datasets if model.lower() in d['model'].lower()]
        
        if dataset:
            filtered_datasets = [d for d in filtered_datasets if dataset.lower() in d['dataset'].lower()]
        
        if n_value is not None:
            n_str = str(n_value)
            filtered_datasets = [d for d in filtered_datasets if d['n_value'] == n_str]
        
        if not filtered_datasets:
            raise ValueError("No datasets match the specified criteria")
        
        latest_dataset = filtered_datasets[0]  # Already sorted by timestamp desc
        print(f"Loading latest dataset: {latest_dataset['directory']}")
        
        return self.load_dataset(latest_dataset['directory'])
    
    def parse_json_columns(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Parse JSON columns (watermark_blocks, decoded_blocks) into structured DataFrames.
        
        Args:
            df: DataFrame containing JSON columns
            
        Returns:
            Dictionary with parsed DataFrames for each JSON column
        """
        parsed_data = {}
        
        json_columns = ['watermark_blocks', 'decoded_blocks']
        
        for col in json_columns:
            if col in df.columns:
                print(f"Parsing {col} column...")
                
                # Parse JSON strings and expand into rows
                rows = []
                for idx, json_str in df[col].items():
                    if pd.notna(json_str) and json_str:
                        try:
                            blocks = json.loads(json_str)
                            for block_idx, block in enumerate(blocks):
                                row = {'original_row_idx': idx, 'block_idx': block_idx}
                                row.update(block)
                                rows.append(row)
                        except json.JSONDecodeError as e:
                            print(f"Warning: Could not parse JSON in row {idx}: {e}")
                
                if rows:
                    parsed_df = pd.DataFrame(rows)
                    parsed_data[col] = parsed_df
                    print(f"  Parsed {len(rows)} blocks from {col}")
                else:
                    print(f"  No valid data found in {col}")
        
        return parsed_data
    
    def get_dataset_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get a summary of the loaded dataset.
        
        Args:
            df: Loaded DataFrame
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_experiments': len(df),
            'unique_prompts': df['prompt'].nunique() if 'prompt' in df.columns else 0,
            'watermark_success_rate': df['watermark_recovered'].mean() if 'watermark_recovered' in df.columns else None,
            'avg_matching_blocks': df['matching_blocks'].mean() if 'matching_blocks' in df.columns else None,
            'avg_tokens_length': df['tokens_length'].mean() if 'tokens_length' in df.columns else None,
            'field_sizes': df['field_size'].unique().tolist() if 'field_size' in df.columns else [],
            'timing_stats': {}
        }
        
        # Add timing statistics if available
        timing_cols = ['encoding_time', 'decoding_time', 'mcp_time']
        for col in timing_cols:
            if col in df.columns:
                summary['timing_stats'][col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max()
                }
        
        return summary


def main():
    """
    Example usage of the WatermarkDataLoader.
    """
    # Initialize the loader
    loader = WatermarkDataLoader()
    
    # List available datasets
    print("Available datasets:")
    available = loader.list_available_datasets()
    print(available.to_string(index=False))
    print()
    
    if len(available) == 0:
        print("No datasets found. Make sure you have run experiments and have data in the output/ directory.")
        return
    
    # Load the latest dataset
    try:
        print("Loading the most recent dataset...")
        df = loader.load_latest_dataset()
        
        # Show basic info
        print(f"\nDataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Get summary
        summary = loader.get_dataset_summary(df)
        print(f"\nDataset Summary:")
        print(f"  Total experiments: {summary['total_experiments']}")
        print(f"  Watermark success rate: {summary['watermark_success_rate']:.2%}" if summary['watermark_success_rate'] is not None else "  Watermark success rate: N/A")
        print(f"  Average matching blocks: {summary['avg_matching_blocks']:.2f}" if summary['avg_matching_blocks'] is not None else "  Average matching blocks: N/A")
        print(f"  Average token length: {summary['avg_tokens_length']:.1f}" if summary['avg_tokens_length'] is not None else "  Average token length: N/A")
        print(f"  Field sizes: {summary['field_sizes']}")
        
        # Parse JSON columns
        print(f"\nParsing JSON columns...")
        parsed_data = loader.parse_json_columns(df)
        
        for col_name, parsed_df in parsed_data.items():
            print(f"  {col_name}: {len(parsed_df)} blocks")
            if len(parsed_df) > 0:
                print(f"    Columns: {list(parsed_df.columns)}")
        
        print(f"\nData loaded successfully! You can now work with:")
        print(f"  - df: Main DataFrame with experiment results")
        print(f"  - parsed_data: Dictionary with parsed JSON columns")
        
        return df, parsed_data
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


if __name__ == "__main__":
    df, parsed_data = main()
