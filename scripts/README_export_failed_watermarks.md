# Export Failed Watermarks Script

This script (`export_failed_watermarks.py`) filters watermarking experiment data for a minimum token count and exports all prompts for failed watermark experiments to a CSV file.

## Features

- Loads data from all available experiment output folders
- Filters experiments by minimum token count (default: 704 tokens)
- Identifies failed watermark experiments (where `watermark_recovered` is False)
- Exports comprehensive data including prompts, generated text, and metadata to CSV
- Provides summary statistics and analysis

## Usage

### Basic Usage
```bash
# Export failed watermarks with default settings (min 704 tokens)
python export_failed_watermarks.py

# Export with custom minimum token count
python export_failed_watermarks.py --min-tokens 500

# Export with custom output filename
python export_failed_watermarks.py --output my_failed_experiments.csv

# Combine options
python export_failed_watermarks.py --min-tokens 800 --output high_token_failures.csv
```

### Command Line Arguments

- `--min-tokens`: Minimum number of tokens to filter for (default: 704)
- `--output`: Output CSV filename (default: auto-generated based on min_tokens)
- `--help`: Show help message

## Output

The script generates a CSV file containing:

### Core Columns
- `prompt`: The input prompt used for generation
- `generated_text`: The watermarked text that was generated
- `token_length`: Number of tokens in the generated text
- `watermark_recovered`: Boolean indicating watermark recovery success (always False in this export)
- `matching_blocks`: Number of watermark blocks that matched during detection

### Metadata Columns
- `model_name`: The LLM model used (e.g., "meta-llama_Llama-2-7b-chat-hf")
- `dataset_name`: The dataset used for prompts (e.g., "ChristophSchuhmann_MS_COCO_2017_URL_TEXT")
- `n_value`: The n-bit encoding parameter used
- `output_folder`: Source experiment folder
- `experiment_timestamp`: When the experiment was run

### Optional Columns (if available)
- `field_size`: Galois field size used
- `encoding_time`: Time taken for watermark encoding
- `decoding_time`: Time taken for watermark detection
- `mcp_time`: Time for additional processing
- `watermark_blocks`: JSON data of encoded watermark blocks
- `decoded_blocks`: JSON data of decoded blocks during detection

## Example Output

The script provides detailed console output including:

1. **Dataset Discovery**: Lists all available experiment folders
2. **Loading Progress**: Shows progress loading each dataset
3. **Filtering Results**: Reports how many experiments meet the criteria
4. **Export Summary**: Statistics about the exported data
5. **Analysis**: Breakdown by model/dataset and token length distribution

## Use Cases

This script is useful for:

- **Failure Analysis**: Understanding why certain watermarks failed to be detected
- **Pattern Recognition**: Identifying common characteristics in failed experiments
- **Model Comparison**: Comparing failure rates across different models
- **Dataset Analysis**: Understanding how different prompt datasets affect watermark success
- **Research**: Providing data for improving watermarking algorithms

## Integration

The script uses the existing `WatermarkDataLoader` class from `load_data.py`, ensuring consistency with other analysis scripts in the project.
