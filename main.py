import argparse
import galois
import secrets
import random
import numpy as np
import torch
import src.paths as paths
from src.utils import get_shuffled_essays
from src.llm_watermark import LLMWatermarkEncoder, LLMWatermarkDecoder
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

    args = parser.parse_args()

    # Set global cache
    paths.set_cache_dir(args.cache_dir)
    paths.ensure_directories()

    # Initialize random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    # Create a Galois field of size 2^n
    field_size = 2 ** args.n
    gf = galois.GF(field_size)
    
    # Generate two random numbers in the Galois field
    a0 = gf.Random()
    a1 = gf.Random()
    if args.verbose:
        print(f"Generated random numbers in GF(2^{args.n}): a0 = {a0}, a1 = {a1}")
        print(f"K = {a0}|{a1}\n")
    
    # Create an anonymous function y = x * a1 + a0
    line_fnc = lambda x: x * a1 + a0
    
    # Generate a 64-bit secret key outside of the Galois field
    secret_key = secrets.token_hex(8)  # 8 bytes = 64 bits, displayed as 16 hexadecimal characters
    if args.verbose:
        print(f"Generated 64-bit secret key: {secret_key}")


    # Determine device
    device = "cpu" if args.no_cuda else None
    
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

    for prompt_idx, prompt in enumerate(prompts, 1):
        if total_prompts > 1:
            print(f"\n{'='*60}")
            print(f"Processing Prompt {prompt_idx}/{total_prompts}")
            if args.verbose:
                print(f"{'='*60}")
                print(f"Prompt preview: {prompt[:100] + '...' if len(prompt) > 100 else prompt}\n")
        
        if args.verbose:
            # Decoding
            print(f"\n{'='*60}")
            print("Encoding")
            print(f"{'='*60}")

        generated_text, statistics, watermark_blocks_info = watermarker.generate_text(
            prompt=prompt,
            max_new_tokens=args.max_tokens,
            verbose=args.verbose
        )
        
        if args.verbose:
            print(f"{'-'*60}")
            print(f"\nPrompt:\n{prompt}")
            print(f"\nGenerated text:\n{generated_text}")
            print(f"\nStatistics: {statistics}")
            print(f"\nWatermark blocks info:")
            # Calculate the maximum width needed for x and y values based on n
            max_gf_value_str_len = len(str(2**args.n - 1))
            for i, block in enumerate(watermark_blocks_info):
                print(f"  Block {i:<3}: x= {block['x']:>{max_gf_value_str_len}}, y= {block['y']:>{max_gf_value_str_len}}, bits= {str(block['y_bits']):<{args.n * 3}}")
            
            # Decoding
            print(f"\n{'='*60}")
            print("Decoding")
            print(f"{'='*60}")
        
        # Create decoder with same parameters
        decoder = LLMWatermarkDecoder(
            model_name=args.model,
            secret_key=secret_key,
            n=args.n,
            gf=gf,
            green_list_fraction=args.green_fraction,
            seed=args.seed,
            verbose=args.verbose
        )
        
        # Decode the generated text
        decoded_blocks = decoder.decode_text(generated_text, prompt)
        
        if args.verbose:
            print(f"Decoded {len(decoded_blocks)} blocks:")
            for i, block in enumerate(decoded_blocks):
                print(f"  Block {i:<3}: x= {block['x']:>{max_gf_value_str_len}}, y_bits= {str(block['y_bits']):<{args.n * 3}}")
            
            # Compare encoder vs decoder results
            print(f"\nComparison (Encoder vs Decoder):")
            print(f"{'Block':<6} {'Encoder X':<12} {'Decoder X':<12} {'Match':<6} {'Encoder Bits':<{args.n * 3}} {'Decoder Bits':<{args.n * 3}} {'Bits Match'}")
            print("-" * (6 + 12 + 12 + 6 + args.n * 3 + args.n * 3 + 10))
            
            for i in range(min(len(watermark_blocks_info), len(decoded_blocks))):
                enc_block = watermark_blocks_info[i]
                dec_block = decoded_blocks[i]
                x_match = "✓" if enc_block['x'] == dec_block['x'] else "✗"
                bits_match = "✓" if enc_block['y_bits'] == dec_block['y_bits'] else "✗"
                
                print(f"{i:<6} {enc_block['x']:<12} {dec_block['x']:<12} {x_match:<6} {str(enc_block['y_bits']):<{args.n * 3}} {str(dec_block['y_bits']):<{args.n * 3}} {bits_match}")


if __name__ == "__main__":
    main()
