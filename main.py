import argparse
import galois
import random
import numpy as np
import torch
import pandas as pd
import json
import os
import time
from datetime import datetime
import src.paths as paths
from src.utils import get_shuffled_essays
from src.llm_watermark import LLMWatermarkEncoder, LLMWatermarkDecoder, MCPSolver
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="LLM Text Watermarking based on Lagrange Interpolation")
    parser.add_argument("--model", type=str, default="facebook/opt-125m", help="Model to use")
    parser.add_argument("--max-tokens", type=int, default=304, help="Maximum tokens to generate")
    parser.add_argument("--green-fraction", type=float, default=0.5, help="Fraction of tokens in green list")
    parser.add_argument("--bias", type=float, default=6.0, help="Bias to add to green/red tokens")
    parser.add_argument("--n", type=int, default=8, help="Size of the field and the blocks to be encoded")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--prompt", type=str, help="Custom prompt (uses random essay from dataset if not provided)")
    parser.add_argument("--dataset", type=str, nargs=4, default=["ChristophSchuhmann/essays-with-instructions", "default", "train", "instructions"], 
                        help="Hugging Face dataset to use for prompts. Format: '<org>/<dataset name>' '<subset>' '<split>' '<column>'. Default: 'ChristophSchuhmann/essays-with-instructions' 'default' 'train' 'instructions'")
    parser.add_argument("--cache-dir", type=str, default=paths.CACHE_DIR, help="Cache directory for models and datasets")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA even if available")
    parser.add_argument("--context-window", type=int, default=1500, help="Maximum number of tokens to use as context for generation (default: 1500)")
    parser.add_argument("--temperature", "--temp", type=float, default=0.0, help="Sampling temperature (default: 0.0 = greedy sampling, higher = more random)")
    parser.add_argument("--hash-window", type=int, default=1, help="Number of previous tokens to hash together (default: 1)") 
    parser.add_argument("--n-prompts", type=int, default=1, help="Number of prompts to process (default: 1)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output and progress information")
    parser.add_argument("--stats", action="store_true", help="Show statistics summary in console (statistics are always saved to file)")
    parser.add_argument("--error-correction-k", type=int, default=0, help="Enable k-bit error correction by generating variants for each decoded block (0=disabled, must be < n)")
    parser.add_argument("--skip-detokenization", action="store_true", help="Use generated token IDs directly for decoding instead of detokenizing and retokenizing text")

    args = parser.parse_args()

    # Validate error correction parameter and show warning
    if args.error_correction_k >= args.n:
        print(f"ERROR: error-correction-k ({args.error_correction_k}) must be less than n ({args.n})")
        return 1
    
    if args.error_correction_k >= 3:
        from math import comb
        num_variants = comb(args.n, args.error_correction_k)
        print(f"WARNING: Using {args.error_correction_k}-bit error correction will generate {num_variants:,} variants per block.")
        print(f"         This may significantly increase memory usage and processing time.")
        print(f"         Consider using smaller k values for large datasets.\n")

    # Set global cache
    paths.set_cache_dir(args.cache_dir)
    paths.ensure_directories()

    # Initialize random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Determine device
    if args.no_cuda:
        device = "cpu"
        print(f"Device: {device} (CUDA disabled by --no-cuda flag)\n")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            print(f"Device: {device} (CUDA available and enabled)\n")
        else:
            print(f"Device: {device} (CUDA not available, falling back to CPU)\n")
        
    # Create a Galois field of size 2^n
    field_size = 2 ** args.n
    gf = galois.GF(field_size)
    
    # Generate two random numbers in the Galois field
    a0 = gf.Random(seed=args.seed)
    a1 = gf.Random(seed=args.seed + 1)
    if args.verbose:
        print(f"Generated random numbers in GF(2^{args.n}): a0 = {a0}, a1 = {a1}")
        print(f"K = {a0}|{a1}\n")
    
    # Create an anonymous function y = x * a1 + a0
    line_fnc = lambda x: x * a1 + a0
    
    # Generate a deterministic 64-bit secret key based on seed
    secret_key_int = random.getrandbits(64)
    secret_key = f"{secret_key_int:016x}"  # Format as 16-character hex string
    if args.verbose:
        print(f"Generated seed based 64-bit secret key: {secret_key}\n")
    
    # Create watermarker instance
    watermarker = LLMWatermarkEncoder(
        model_name=args.model,
        secret_key=secret_key,
        line_fnc=line_fnc,
        n=args.n,
        gf=gf,
        green_list_fraction=args.green_fraction,
        bias=args.bias,
        seed=args.seed,
        device=device,
        context_window=args.context_window,
        temperature=args.temperature,
        hash_window=args.hash_window,
        verbose=args.verbose
    )

    # Get dataset info
    dataset_name = args.dataset[0]
    dataset_subset = args.dataset[1]
    dataset_split = args.dataset[2]
    dataset_column = args.dataset[3]

    # Get prompts
    if args.prompt:
        # Single custom prompt
        prompts = [args.prompt]
        if args.verbose:
            print(f"\n--- Using Custom Prompt ---")
            print(args.prompt[:200] + "..." if len(args.prompt) > 200 else args.prompt)
            print("---------------------------\n")

    else:
        # Get shuffled essays from dataset
        prompts = get_shuffled_essays(
            dataset_name=dataset_name,
            dataset_subset=dataset_subset,
            dataset_split=dataset_split,
            dataset_column=dataset_column,
            seed=args.seed,
            n_prompts=args.n_prompts
        )
        if args.verbose:
            print(f"\n--- Prompt from {dataset_name}/{dataset_subset} ({dataset_split} split, {dataset_column} column) ---")
            print(prompts[0][:200] + "..." if len(prompts[0]) > 200 else prompts[0])
            print(f"{'='*60}\n")
    
    total_prompts = len(prompts)
    if args.verbose:
        print(f"Generating {args.max_tokens} tokens per prompt with watermarking...")
        print(f"Using dataset: {dataset_name}")
        print(f"Processing {total_prompts} prompt(s) with model: {args.model}")

    # Create dynamic output directory and file path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_clean = args.model.replace("/", "_")
    dataset_name_clean = dataset_name.replace("/", "_")
    
    output_subdir = f"{model_name_clean}_{dataset_name_clean}_n{args.n}_{timestamp}"
    output_dir = os.path.join("output", output_subdir)
    os.makedirs(output_dir, exist_ok=True)
    
    stats_file = os.path.join(output_dir, "statistics.csv")
    stats_parquet_file = os.path.join(output_dir, "statistics.parquet")
    
    # Initialize DataFrame with all required columns
    columns = [
        'field_size', 'prompt', 'generated_text', 'tokens_length', 
        'a0', 'a1', 'recovered_a0', 'recovered_a1', 'secret_key',
        'watermark_blocks', 'decoded_blocks', 'matching_blocks', 
        'watermark_recovered', 'encoding_time', 'decoding_time', 'mcp_time'
    ]
    
    # Define explicit column types for proper data handling
    column_dtypes = {
        'field_size': 'int64',
        'prompt': 'string',
        'generated_text': 'string', 
        'tokens_length': 'int64',
        'a0': 'int64',
        'a1': 'int64',
        'recovered_a0': 'Int64',  # Nullable integer for None values
        'recovered_a1': 'Int64',  # Nullable integer for None values
        'secret_key': 'string',
        'watermark_blocks': 'string',  # JSON string
        'decoded_blocks': 'string',    # JSON string
        'matching_blocks': 'int64',
        'watermark_recovered': 'boolean',
        'encoding_time': 'float64',
        'decoding_time': 'float64',
        'mcp_time': 'float64'
    }
    
    stats_df = pd.DataFrame(index=range(total_prompts), columns=columns)

    for prompt_idx, prompt in enumerate(prompts, 0):
        if total_prompts > 1:
            print(f"\n{'='*60}")
            print(f"Processing Prompt {prompt_idx+1}/{total_prompts}")
            if args.verbose:
                print(f"{'='*60}")
                print(f"Prompt preview: {prompt[:100] + '...' if len(prompt) > 100 else prompt}\n")
        
        if args.verbose:
            # Encoding
            print(f"\n{'-'*60}")
            print("Encoding")
            print(f"{'-'*60}")

        # Time encoding
        encoding_start = time.time()
        full_text, generated_text, generated_ids, formatted_prompt, generation_statistics, watermark_blocks_info = watermarker.generate_text(
            prompt=prompt,
            max_new_tokens=args.max_tokens,
            verbose=args.verbose
        )
        encoding_time = time.time() - encoding_start
        
        if args.verbose:
            print(f"{'-'*60}")
            print(f"\nPrompt:\n{prompt}")
            print(f"\nGenerated text:\n{generated_text}")

            print(f"\n{'-'*60}")

            print(f"\nGeneration statistics:")
            print(f"  Total tokens: {generation_statistics['total_tokens_generated']}")
            print(f"  Green tokens: {generation_statistics['green_tokens']}")
            print(f"  Red tokens: {generation_statistics['red_tokens']}")
            print(f"  Greeen ratio: {generation_statistics['green_ratio']}")
            print(f"  Blocks encoded: {generation_statistics['blocks_encoded']}")

            print(f"\nWatermark blocks info:")
            # Calculate the maximum width needed for x and y values based on n
            max_gf_value_str_len = len(str(2**args.n - 1))
            for i, block in enumerate(watermark_blocks_info):
                print(f"  Block {i+1:<3}: x= {block['x']:>{max_gf_value_str_len}}, y= {block['y']:>{max_gf_value_str_len}}, bits= {str(block['y_bits']):<{args.n * 3}}")
            
            # Decoding
            print(f"\n{'-'*60}")
            print("Decoding")
            print(f"{'-'*60}")
        
        # Create decoder with same parameters
        decoder = LLMWatermarkDecoder(
            model_name=args.model,
            secret_key=secret_key,
            n=args.n,
            gf=gf,
            green_list_fraction=args.green_fraction,
            seed=args.seed,
            device=device,
            verbose=args.verbose,
            error_correction_k=args.error_correction_k
        )
        
        # Time decoding
        decoding_start = time.time()
        if args.skip_detokenization:
            decoded_blocks, decoded_tokens_length = decoder.decode_text(generated_ids=generated_ids)
        else:
            decoded_blocks, decoded_tokens_length = decoder.decode_text(generated_text=generated_text)
        decoding_time = time.time() - decoding_start
        
        if args.verbose:
            print(f"Decoded {len(decoded_blocks)} blocks:")
            for i, block in enumerate(decoded_blocks):
                print(f"  Block {i+1:<3}: x= {block['x']:>{max_gf_value_str_len}}, y_bits= {str(block['y_bits']):<{args.n * 3}}")
            
            # Compare encoder vs decoder results
            print(f"\nComparison (Watermark vs Encoded vs Decoded):")
            print(f"{'Block':<6} {'Enc X':<6} {'Dec X':<6} {'Watermark Bits':<{args.n * 3 + 2}} {'Encoder Bits':<{args.n * 3 + 2}} {'Decoder Bits':<{args.n * 3 + 2}} {'X / Decoding / Watermark Matches'}")
            print("-" * (18 + args.n * 3 + 2 + args.n * 3 + 2 + args.n * 3 + 35))
            
            for i in range(min(len(watermark_blocks_info), len(decoded_blocks))):
                enc_block = watermark_blocks_info[i]
                dec_block = decoded_blocks[i]
                
                # Check if X coordinates match
                x_match = "✓" if enc_block['x'] == dec_block['x'] else "✗"
                
                # Check if encoded bits match decoded bits (encoding/decoding consistency)
                decoding_match = "✓" if enc_block['encoded_bits'] == dec_block['y_bits'] else "✗"
                
                # Check if watermark bits match decoded bits (watermark recovery success)
                watermark_match = "✓" if enc_block['y_bits'] == dec_block['y_bits'] else "✗"
                
                print(f"{i+1:<6} {enc_block['x']:<6} {dec_block['x']:<6} {str(enc_block['y_bits']):<{args.n * 3 + 2}} {str(enc_block['encoded_bits']):<{args.n * 3 + 2}} {str(dec_block['y_bits']):<{args.n * 3 + 2}} {x_match:<4} {decoding_match:<8} {watermark_match}")
            
            # MCP Verification
            print(f"\n{'-'*60}")
            print("MCP Watermark Verification")
            print(f"{'-'*60}")
        
        # Create MCP solver
        mcp_solver = MCPSolver(gf=gf, n=args.n, verbose=args.verbose)
        
        # Time MCP verification
        mcp_start = time.time()
        verification_result = mcp_solver.verify_watermark(decoded_blocks, a0, a1, watermark_blocks_info)
        mcp_time = time.time() - mcp_start
        
        if args.verbose:
            print(f"Verification Results:")
            print(f"  Collinear Points: {verification_result['max_collinear_count']}/{verification_result['total_points']}")
            print(f"  Original a₀: {a0}")
            print(f"  Recovered a₀: {verification_result['recovered_a0']}")
            print(f"  Original a₁: {a1}")
            print(f"  Recovered a₁: {verification_result['recovered_a1']}")
            print(f"  Matching blocks: {verification_result['matching_blocks']}")
            
            if verification_result['is_valid']:
                print(f"✅ Watermark successfully verified!")
            else:
                print(f"❌ Watermark verification failed!")
        else:
            # Compact output for non-verbose mode
            status = "✅ VALID" if verification_result['is_valid'] else "❌ INVALID"
            print(f"Watermark Verification: {status} (Collinear points: {verification_result['max_collinear_count']}/{verification_result['total_points']}, Matching: {verification_result['matching_blocks']})")

        # Collect statistics for this prompt run
        stats_df.loc[prompt_idx, 'field_size'] = 2 ** args.n
        stats_df.loc[prompt_idx, 'prompt'] = prompt
        stats_df.loc[prompt_idx, 'generated_text'] = generated_text
        stats_df.loc[prompt_idx, 'tokens_length'] = decoded_tokens_length
        stats_df.loc[prompt_idx, 'a0'] = int(a0)
        stats_df.loc[prompt_idx, 'a1'] = int(a1)
        stats_df.loc[prompt_idx, 'recovered_a0'] = int(verification_result['recovered_a0']) if verification_result['recovered_a0'] is not None else None
        stats_df.loc[prompt_idx, 'recovered_a1'] = int(verification_result['recovered_a1']) if verification_result['recovered_a1'] is not None else None
        stats_df.loc[prompt_idx, 'secret_key'] = secret_key
        stats_df.loc[prompt_idx, 'watermark_blocks'] = json.dumps(watermark_blocks_info)
        stats_df.loc[prompt_idx, 'decoded_blocks'] = json.dumps(decoded_blocks)
        stats_df.loc[prompt_idx, 'matching_blocks'] = verification_result['matching_blocks']
        stats_df.loc[prompt_idx, 'watermark_recovered'] = verification_result['is_valid']
        stats_df.loc[prompt_idx, 'encoding_time'] = encoding_time
        stats_df.loc[prompt_idx, 'decoding_time'] = decoding_time
        stats_df.loc[prompt_idx, 'mcp_time'] = mcp_time

    # Apply proper data types to all columns
    for col, dtype in column_dtypes.items():
        if col in stats_df.columns:
            try:
                stats_df[col] = stats_df[col].astype(dtype)
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not convert column '{col}' to {dtype}: {e}")
                # Keep original type if conversion fails
    
    # Export statistics to both CSV and Parquet formats
    stats_df.to_csv(stats_file, index=False)
    stats_df.to_parquet(stats_parquet_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"Statistics saved to:")
    print(f"  CSV: {stats_file} ({os.path.getsize(stats_file):,} bytes)")
    print(f"  Parquet: {stats_parquet_file} ({os.path.getsize(stats_parquet_file):,} bytes)")
    
    if args.stats:
        print(f"\nStatistics Summary:")
        print(f"  Total prompts processed: {total_prompts}")
        print(f"  Successful watermark recoveries: {stats_df['watermark_recovered'].sum()}")
        print(f"  Watermark success rate: {stats_df['watermark_recovered'].mean():.2%}")
        print(f"  Average matching blocks: {stats_df['matching_blocks'].mean():.2f}")
        print(f"  Average timing:")
        print(f"  Encoding: {stats_df['encoding_time'].mean():.3f}s")
        print(f"  Decoding: {stats_df['decoding_time'].mean():.3f}s")
        print(f"  MCP: {stats_df['mcp_time'].mean():.3f}s\n")


if __name__ == "__main__":
    main()
