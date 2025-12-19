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
from src.pm_galois import GaloisField
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
    parser.add_argument("--n-prompts", type=str, default="1", help="Number of prompts to process, or 'all' for entire dataset (default: 1)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output and progress information")
    parser.add_argument("--stats", action="store_true", help="Show statistics summary in console (statistics are always saved to file)")
    parser.add_argument("--force-tokenization", action="store_true", help="Force detokenization and retokenization of text for decoding instead of using token IDs directly")
    parser.add_argument("--hamming", type=str, choices=["none", "standard", "secded"], default="none",
                        help="Hamming code mode: 'none' (default), 'standard' (correct 1-bit errors), 'secded' (correct 1-bit, detect 2-bit errors)")
    parser.add_argument("--correct", action="store_true", default=False,
                        help="Enable Hamming error correction (default: detection-only for better filtering)")
    parser.add_argument("--c-correction", type=int, default=0,
                        help="Enable c-correction mode: generate all bit-flip variations up to this Hamming distance (e.g., 1 or 2). Incompatible with --hamming and --correct.")

    args = parser.parse_args()

    # Parse n_prompts: accept "all" or a positive integer
    if args.n_prompts.lower() == "all":
        n_prompts = None
    else:
        try:
            n_prompts = int(args.n_prompts)
            if n_prompts < 1:
                print(f"ERROR: --n-prompts must be a positive integer or 'all', got: {args.n_prompts}")
                return 1
        except ValueError:
            print(f"ERROR: --n-prompts must be a positive integer or 'all', got: {args.n_prompts}")
            return 1

    # Validate c-correction incompatibility with hamming and correct
    if args.c_correction > 0:
        if args.hamming != "none":
            print(f"ERROR: --c-correction is incompatible with --hamming {args.hamming}. Use --hamming none (default) with --c-correction.")
            return 1
        if args.correct:
            print(f"ERROR: --c-correction is incompatible with --correct.")
            return 1
        if args.c_correction > 2:
            print(f"WARNING: --c-correction > 2 is not recommended. High values generate many variations, making MCP calculation slower and more prone to errors.")

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
        hamming_mode=args.hamming,
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
            n_prompts=n_prompts
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
        if args.hamming != "none":
            print(f"Hamming mode: {args.hamming}")
            print(f"  Data bits per block: {args.n}")
            print(f"  Parity bits: {watermarker.hamming.parity_bit_count}")
            print(f"  Tokens per block: {watermarker.tokens_per_block}")

    # Create dynamic output directory and file path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_clean = args.model.replace("/", "_")
    dataset_name_clean = dataset_name.replace("/", "_")
    
    output_subdir = f"{model_name_clean}_{dataset_name_clean}_n{args.n}_{timestamp}"
    output_dir = os.path.join("output", output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    # Save run configuration
    run_config = {
        'model': args.model,
        'max_tokens': args.max_tokens,
        'green_fraction': args.green_fraction,
        'bias': args.bias,
        'n': args.n,
        'seed': args.seed,
        'dataset': args.dataset,
        'context_window': args.context_window,
        'temperature': args.temperature,
        'hash_window': args.hash_window,
        'n_prompts': args.n_prompts,
        'force_tokenization': args.force_tokenization,
        'hamming': args.hamming,
        'correct': args.correct,
        'c_correction': args.c_correction,
        'timestamp': timestamp,
        'output_dir': output_subdir
    }
    config_file = os.path.join(output_dir, "run_config.json")
    with open(config_file, 'w') as f:
        json.dump(run_config, f, indent=2)

    stats_file = os.path.join(output_dir, "statistics.csv")
    stats_parquet_file = os.path.join(output_dir, "statistics.parquet")
    
    # Initialize DataFrame with all required columns
    columns = [
        'field_size', 'prompt', 'generated_text', 'generated_ids', 'tokens_length',
        'a0', 'a1', 'recovered_a0', 'recovered_a1', 'secret_key',
        'watermark_blocks', 'encoded_blocks', 'decoded_blocks', 'valid_blocks', 'matching_blocks',
        'unique_watermark_blocks_count', 'unique_valid_blocks_count', 'unique_matching_blocks_count',
        'properly_encoded_tokens', 'watermark_recovered', 'encoding_time', 'decoding_time', 'mcp_time'
    ]

    # Define explicit column types for proper data handling
    column_dtypes = {
        'field_size': 'int64',
        'prompt': 'string',
        'generated_text': 'string',
        'generated_ids': 'string',     # JSON string
        'tokens_length': 'int64',
        'a0': 'int64',
        'a1': 'int64',
        'recovered_a0': 'Int64',       # Nullable integer for None values
        'recovered_a1': 'Int64',       # Nullable integer for None values
        'secret_key': 'string',
        'watermark_blocks': 'string',  # JSON string (intended watermark)
        'encoded_blocks': 'string',    # JSON string (actually encoded)
        'decoded_blocks': 'string',    # JSON string (all blocks)
        'valid_blocks': 'string',      # JSON string (valid blocks only)
        'matching_blocks': 'string',   # JSON string (blocks matching watermark)
        'unique_watermark_blocks_count': 'int64',
        'unique_valid_blocks_count': 'int64',
        'unique_matching_blocks_count': 'int64',
        'properly_encoded_tokens': 'int64',
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
        full_text, generated_text, generated_ids, formatted_prompt, generation_statistics, watermark_blocks, encoded_blocks = watermarker.generate_text(
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

            # Count unique (x, y) pairs
            unique_pairs = len(set((b['x'], b['y']) for b in watermark_blocks))
            print(f"\nWatermark blocks info ({unique_pairs}/{len(watermark_blocks)} unique):")
            # Calculate the maximum width needed for x and y values based on n
            max_gf_value_str_len = len(str(2**args.n - 1))
            for i, block in enumerate(watermark_blocks):
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
            hamming_mode=args.hamming,
            correct=args.correct,
            c_correction=args.c_correction
        )
        
        # Time decoding
        decoding_start = time.time()
        if args.force_tokenization:
            all_blocks, valid_blocks, decoded_tokens_length = decoder.decode_text(generated_text=generated_text)
        else:
            all_blocks, valid_blocks, decoded_tokens_length = decoder.decode_text(generated_ids=generated_ids)
        decoding_time = time.time() - decoding_start
        
        if args.verbose:
            if args.hamming != "none" or args.c_correction > 0:
                # Hamming/c-correction mode: Show recovery table with y-value matching
                if args.hamming != "none":
                    correction_mode = "on" if args.correct else "off"
                    print(f"Sliding window: {len(all_blocks)} scanned, {len(valid_blocks)} valid, {len(all_blocks) - len(valid_blocks)} invalid (correction={correction_mode})")
                else:
                    print(f"C-correction: {len(valid_blocks)} blocks ({len(valid_blocks) // (1 + args.n)} original, expanded with c={args.c_correction})")

                print(f"\nWatermark Recovery:")
                print(f"  {'#':>3}  {'Encoded (x, y)':<17}  {'Decoded (x, y)':<17}  Status")
                print(f"  {'--':>3}  {'-'*17}  {'-'*17}  {'-'*10}")

                recovered = 0
                recovered_pairs = set()  # Track unique (x, y) pairs recovered
                decoded_y_map = {b['y']: b for b in valid_blocks}

                for i, wm in enumerate(watermark_blocks, 1):
                    wm_coord = f"({wm['x']}, {wm['y']})"
                    if wm['y'] in decoded_y_map:
                        dec = decoded_y_map[wm['y']]
                        dec_coord = f"({dec['x']}, {dec['y']})"
                        recovered_pairs.add((dec['x'], dec['y']))
                        if dec['x'] == wm['x']:
                            status = "RECOVERED"
                        else:
                            status = "RECOVERED*"  # y matched but at wrong x position
                        recovered += 1
                    else:
                        dec_coord = "-"
                        status = "MISSING"
                    print(f"  {i:>3}  {wm_coord:<17}  {dec_coord:<17}  {status}")

                noise = len(valid_blocks) - recovered
                recovery_pct = 100 * recovered / len(watermark_blocks) if watermark_blocks else 0
                unique_recovered = len(recovered_pairs)
                print(f"\n  Recovery: {recovered}/{len(watermark_blocks)} ({recovery_pct:.1f}%), {unique_recovered} unique")
                print(f"  Noise: {noise} extra valid blocks")
            else:
                # Non-Hamming mode: Original output
                print(f"Decoded {len(all_blocks)} total blocks, {len(valid_blocks)} valid blocks:")
                for i, block in enumerate(valid_blocks):
                    print(f"  Block {i+1:<3}: x= {block['x']:>{max_gf_value_str_len}}, y_bits= {str(block['y_bits']):<{args.n * 3}}, p_bits= {block['p_bits']}")

                # Compare encoder vs decoder results (only for non-Hamming fixed-boundary mode)
                print(f"\nComparison (Watermark vs Encoded vs Decoded):")
                print(f"{'Block':<6} {'Enc X':<6} {'Dec X':<6} {'Watermark Bits':<{args.n * 3 + 2}} {'Encoder Bits':<{args.n * 3 + 2}} {'Decoder Bits':<{args.n * 3 + 2}} {'X / Decoding / Watermark'}")
                print("-" * (18 + args.n * 3 + 2 + args.n * 3 + 2 + args.n * 3 + 35))

                for i in range(min(len(watermark_blocks), len(valid_blocks))):
                    wm_block = watermark_blocks[i]
                    enc_block = encoded_blocks[i]
                    dec_block = valid_blocks[i]

                    # Check if X coordinates match
                    x_match = "Y" if wm_block['x'] == dec_block['x'] else "N"

                    # Check if encoded bits match decoded bits (encoding/decoding consistency)
                    decoding_match = "Y" if enc_block['y_bits'] == dec_block['y_bits'] else "N"

                    # Check if watermark bits match decoded bits (watermark recovery success)
                    watermark_match = "Y" if wm_block['y_bits'] == dec_block['y_bits'] else "N"

                    print(f"{i+1:<6} {wm_block['x']:<6} {dec_block['x']:<6} {str(wm_block['y_bits']):<{args.n * 3 + 2}} {str(enc_block['y_bits']):<{args.n * 3 + 2}} {str(dec_block['y_bits']):<{args.n * 3 + 2}} {x_match:<4} {decoding_match:<10} {watermark_match}")
            
            # MCP Verification
            print(f"\n{'-'*60}")
            print("MCP Watermark Verification")
            print(f"{'-'*60}")
        
        # Create MCP solver
        # Note: MCPSolver uses pm_galois.GaloisField (has subtract/divide methods),
        # not galois.GF which is used by encoder/decoder for field arithmetic
        pm_gf = GaloisField(args.n)
        mcp_solver = MCPSolver(gf=pm_gf, n=args.n, verbose=args.verbose)

        # Time MCP verification (use valid_blocks for verification)
        mcp_start = time.time()
        verification_result = mcp_solver.verify_watermark(valid_blocks, a0, a1, watermark_blocks)
        mcp_time = time.time() - mcp_start
        
        if args.verbose:
            print(f"Verification Results:")
            print(f"  Collinear Points: {verification_result['max_collinear_count']}/{verification_result['total_points']}")
            print(f"  Original a0: {a0}")
            print(f"  Recovered a0: {verification_result['recovered_a0']}")
            print(f"  Original a1: {a1}")
            print(f"  Recovered a1: {verification_result['recovered_a1']}")
            print(f"  Matching blocks: {len(verification_result['matching_blocks'])}")
            
            if verification_result['is_valid']:
                print(f"Watermark successfully verified!")
            else:
                print(f"Watermark verification failed!")
        else:
            # Compact output for non-verbose mode
            status = "VALID" if verification_result['is_valid'] else "INVALID"
            matching_count = len(verification_result['matching_blocks'])
            print(f"Watermark Verification: {status} (Collinear points: {verification_result['max_collinear_count']}/{verification_result['total_points']}, Matching: {matching_count})")

        # Collect statistics for this prompt run
        # Block types:
        # - watermark_blocks: the (x, y) points we intended to encode (on the secret line)
        # - encoded_blocks: the (x, y) points actually encoded (may differ due to biasing limitations)
        # - decoded_blocks (all_blocks): blocks recovered through decoding (includes c-correction variations)
        # - valid_blocks: for --hamming: Hamming-valid blocks only; otherwise same as decoded_blocks
        # - matching_blocks: valid_blocks whose y-value matches any watermark_block's y-value
        stats_df.loc[prompt_idx, 'field_size'] = 2 ** args.n
        stats_df.loc[prompt_idx, 'prompt'] = prompt
        stats_df.loc[prompt_idx, 'generated_text'] = generated_text
        stats_df.loc[prompt_idx, 'generated_ids'] = json.dumps(generated_ids)
        stats_df.loc[prompt_idx, 'tokens_length'] = decoded_tokens_length
        stats_df.loc[prompt_idx, 'a0'] = int(a0)
        stats_df.loc[prompt_idx, 'a1'] = int(a1)
        stats_df.loc[prompt_idx, 'recovered_a0'] = int(verification_result['recovered_a0']) if verification_result['recovered_a0'] is not None else None
        stats_df.loc[prompt_idx, 'recovered_a1'] = int(verification_result['recovered_a1']) if verification_result['recovered_a1'] is not None else None
        stats_df.loc[prompt_idx, 'secret_key'] = secret_key
        stats_df.loc[prompt_idx, 'watermark_blocks'] = json.dumps(watermark_blocks)
        stats_df.loc[prompt_idx, 'encoded_blocks'] = json.dumps(encoded_blocks)
        stats_df.loc[prompt_idx, 'decoded_blocks'] = json.dumps(all_blocks)
        stats_df.loc[prompt_idx, 'valid_blocks'] = json.dumps(valid_blocks)
        stats_df.loc[prompt_idx, 'matching_blocks'] = json.dumps(verification_result['matching_blocks'])
        stats_df.loc[prompt_idx, 'unique_watermark_blocks_count'] = len(set((b['x'], b['y']) for b in watermark_blocks))
        stats_df.loc[prompt_idx, 'unique_valid_blocks_count'] = len(set((b['x'], b['y']) for b in valid_blocks))
        stats_df.loc[prompt_idx, 'unique_matching_blocks_count'] = len(set((b['x'], b['y']) for b in verification_result['matching_blocks']))
        stats_df.loc[prompt_idx, 'properly_encoded_tokens'] = generation_statistics['properly_encoded_tokens']
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
        # Calculate total properly encoded tokens percentage across all prompts
        total_properly_encoded = stats_df['properly_encoded_tokens'].sum()
        total_tokens_generated = stats_df['tokens_length'].sum()
        properly_encoded_percentage = (total_properly_encoded / total_tokens_generated) * 100 if total_tokens_generated > 0 else 0

        # Calculate average matching blocks from JSON
        matching_counts = stats_df['matching_blocks'].apply(lambda x: len(json.loads(x)) if x else 0)
        avg_matching_blocks = matching_counts.mean()

        print(f"\nStatistics Summary:")
        print(f"  Total prompts processed: {total_prompts}")
        print(f"  Successful watermark recoveries: {stats_df['watermark_recovered'].sum()}")
        print(f"  Watermark success rate: {stats_df['watermark_recovered'].mean():.2%}")
        print(f"  Average matching blocks: {avg_matching_blocks:.2f}")
        print(f"  Total properly encoded tokens: {total_properly_encoded}/{total_tokens_generated} ({properly_encoded_percentage:.2f}%)")
        print(f"  Average timing:")
        print(f"    Encoding: {stats_df['encoding_time'].mean():.3f}s")
        print(f"    Decoding: {stats_df['decoding_time'].mean():.3f}s")
        print(f"    MCP: {stats_df['mcp_time'].mean():.3f}s\n")


if __name__ == "__main__":
    main()
