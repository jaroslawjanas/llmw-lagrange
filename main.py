import argparse
import galois
import secrets
import src.paths as paths
import random
import numpy as np
import torch

def main():
    parser = argparse.ArgumentParser(description="LLM Text Watermarking based on Lagrange Interpolation")
    parser.add_argument("--model", type=str, default="facebook/opt-125m", help="Model to use")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens to generate")
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
    print(f"Generated random numbers in GF(2^{args.n}): a0 = {a0}, a1 = {a1}")
    print(f"K = {a0}|{a1}")
    
    # Create an anonymous function y = x * a1 + a0
    line_fnc = lambda x: x * a1 + a0
    
    # Generate a 64-bit secret key outside of the Galois field
    secret_key = secrets.token_hex(8)  # 8 bytes = 64 bits, displayed as 16 hexadecimal characters
    print(f"Generated 64-bit secret key: {secret_key}")


    # Determine device
    device = "cpu" if args.no_cuda else None

    # Import and test the watermarker
    from src.llm_watermark import LLMWatermarker
    
    # Create watermarker instance
    watermarker = LLMWatermarker(
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
        hash_window=args.hash_window
    )
    
    # Test with a simple prompt
    test_prompt = args.prompt if args.prompt else "The quick brown fox"
    print(f"\nGenerating text with watermark for prompt: '{test_prompt}'")
    
    generated_text, statistics, watermark_blocks = watermarker.generate_text(
        prompt=test_prompt,
        max_new_tokens=args.max_tokens,
        verbose=True
    )
    
    print(f"\nGenerated text:\n{generated_text}")
    print(f"\nStatistics: {statistics}")
    print(f"\nWatermark blocks info:")
    for i, block in enumerate(watermark_blocks):
        print(f"  Block {i}: x={block['x']}, y={block['y']}, bits={block['y_bits']}")


if __name__ == "__main__":
    main()
