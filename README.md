# LLMW-Lagrange: LLM Text Watermarking with Lagrange Interpolation

A robust implementation of LLM text watermarking based on Lagrange interpolation over Galois fields. This system embeds multi-bit watermarks in AI-generated text, enabling traceability and authenticity verification even under adversarial manipulation.

**Based on:** [LLM-Text Watermarking based on Lagrange Interpolation](https://arxiv.org/abs/2505.05712)

## Overview

This project implements a multi-bit watermarking technique for large language models using Lagrange interpolation. The watermark is embedded by biasing token selection during generation to encode points on a secret line equation `f(x) = a₀ + a₁x` over GF(2ⁿ). Verification uses a Maximum Collinear Points (MCP) algorithm to recover the secret line from generated text.

### How It Works

1. **Encoding**: During text generation, each block of n tokens encodes one point (x, y) on the secret line. The x-coordinate is derived by hashing the previous token with a secret key, and y = f(x) is converted to binary. Each bit biases token selection toward "green" (1) or "red" (0) vocabulary partitions.

2. **Decoding**: Tokens are classified as green/red based on the same deterministic vocabulary split. Binary sequences are reconstructed into (x, y) points.

3. **Verification**: The MCP algorithm finds the maximum set of collinear points. If the recovered line coefficients match the original secret, the watermark is verified.

## Key Features

- **Multi-bit Watermarking**: Embeds n-bit blocks using Lagrange interpolation over Galois fields GF(2ⁿ)
- **Configurable Field Sizes**: Supports n from 1 to 20 (field sizes from 2 to 2²⁰)
- **Error Detection/Correction**:
  - Standard Hamming codes (correct 1-bit errors)
  - SECDED mode (correct 1-bit, detect 2-bit errors)
  - C-correction mode (brute-force bit-flip recovery)
- **Model Agnostic**: Works with any HuggingFace transformer model
- **Batch Processing**: Process multiple prompts from datasets
- **Attack Simulation**: Test watermark robustness against insertion, deletion, and substitution attacks
- **Multiprocessing Support**: Parallel attack simulation for faster testing

## Installation

1. Clone the repository:
```bash
git clone https://github.com/jaroslawjanas/llmw-lagrange.git
cd llmw-lagrange
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate llmw-lagrange
```

3. (Optional) Set up HuggingFace token for restricted models:
```bash
cp hf_token.template hf_token
# Edit hf_token with your HuggingFace token
```

## Quick Start

### Basic Usage

Generate watermarked text with default settings:

```bash
python main.py --prompt "Write a story about artificial intelligence"
```

### Advanced Configuration

```bash
python main.py \
    --model "meta-llama/Llama-2-7b-hf" \
    --prompt "Explain quantum computing" \
    --max-tokens 256 \
    --n 8 \
    --bias 6.0 \
    --temperature 0.7 \
    --verbose \
    --stats
```

### Batch Processing

Process multiple prompts from a dataset:

```bash
python main.py \
    --dataset "ChristophSchuhmann/essays-with-instructions" "default" "train" "instructions" \
    --n-prompts 10 \
    --max-tokens 304 \
    --stats
```

### With Hamming Error Detection

```bash
python main.py \
    --prompt "Your prompt here" \
    --hamming standard \
    --max-tokens 204  # Must account for parity bits
```

### With Error Correction

```bash
# Hamming correction mode
python main.py --prompt "Your prompt" --hamming secded --correct

# C-correction mode (brute-force bit flips)
python main.py --prompt "Your prompt" --c-correction 1
```

## Command Line Arguments

### Core Parameters

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | HuggingFace model identifier | `facebook/opt-125m` |
| `--max-tokens` | Maximum tokens to generate (must be divisible by tokens_per_block) | 304 |
| `--n` | Galois field size parameter GF(2ⁿ) and block size | 8 |
| `--prompt` | Custom text prompt | - |
| `--seed` | Random seed for reproducibility | 42 |

### Watermarking Parameters

| Argument | Description | Default |
|----------|-------------|---------|
| `--green-fraction` | Fraction of vocabulary in green list | 0.5 |
| `--bias` | Logit bias for watermark tokens | 6.0 |

### Generation Parameters

| Argument | Description | Default |
|----------|-------------|---------|
| `--temperature` | Sampling temperature (0 = greedy) | 0.0 |
| `--context-window` | Maximum context length | 1500 |
| `--hash-window` | Previous tokens for hashing | 1 |

### Dataset Parameters

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | HuggingFace dataset (4 args: name subset split column) | essays-with-instructions |
| `--n-prompts` | Number of prompts to process (or "all") | 1 |

### Error Handling Parameters

| Argument | Description | Default |
|----------|-------------|---------|
| `--hamming` | Hamming code mode: "none", "standard", or "secded" | none |
| `--correct` | Enable Hamming error correction (default: detection-only) | false |
| `--c-correction` | Enable c-correction: generate all bit-flip variations up to this Hamming distance | 0 |

### System Parameters

| Argument | Description | Default |
|----------|-------------|---------|
| `--cache-dir` | Cache directory for models/datasets | ./cache |
| `--no-cuda` | Disable CUDA acceleration | false |
| `--verbose` | Show detailed output | false |
| `--stats` | Display statistics summary | false |
| `--force-tokenization` | Force detokenization and retokenization (not recommended) | false |

## Architecture

### Core Components

```
src/
├── llm_watermark.py     # Encoder, decoder, MCP solver
├── hamming.py           # Hamming code implementation (systematic format)
├── pm_galois.py         # Optimized GF(2^n) with precomputed inverses
├── model_formatters.py  # Model-specific prompt formatting
├── utils.py             # Dataset loading, HuggingFace token handling
└── paths.py             # Cache directory management

scripts/
├── lib/
│   ├── __init__.py      # Package exports
│   ├── loader.py        # ExperimentLoader class
│   └── data_utils.py    # Shared data loading utilities
├── analyze.py           # Experiment analysis with statistics and plots
└── attack_simulation.py # Attack robustness testing
```

### Pipeline Classes

#### LLMWatermarkEncoder
- Generates watermarked text with K/V cache optimization
- Biases token selection based on secret line equation
- Supports Hamming code encoding
- Tracks encoding statistics (green/red tokens, properly encoded ratio)

#### LLMWatermarkDecoder
- Three decoding modes:
  - **Fixed blocks**: Standard n-token boundary decoding
  - **Sliding window**: Hamming-validity scanning (for `--hamming`)
  - **C-correction**: Brute-force bit-flip variations (for `--c-correction`)
- Uses same vocabulary split as encoder

#### MCPSolver
- O(N²) algorithm for Maximum Collinear Points
- Uses optimized Galois field implementation (pm_galois.py)
- Recovers line equation from collinear points

#### HammingCode
- Systematic format: `[data_bits | parity_bits]`
- Standard mode: d_min=3, correct 1-bit errors
- SECDED mode: d_min=4, correct 1-bit, detect 2-bit errors
- Detection-only mode for better filtering (lower false positives)

### Block Structure

Each watermark block encodes:
- `x`: X-coordinate from hash(previous_token + secret_key) mod 2ⁿ
- `y`: Y-value from f(x) = a₀ + a₁x
- `y_bits`: Binary representation of y (n bits)
- `p_bits`: Parity bits (for Hamming mode)

**Tokens per block:**
- Standard mode: n tokens
- Hamming mode: n + parity_bits tokens (e.g., n=8 → 12 tokens)

### Watermarking Process Flow

```
Encoding:
  previous_token → SHA-256 hash → x ∈ GF(2ⁿ)
  y = a₀ + a₁x → n-bit binary → [optional: Hamming encode]
  For each bit: bias vocabulary → generate token

Decoding:
  tokens → green/red classification → bit sequence
  [optional: Hamming decode or c-correction]
  → reconstruct (x, y) points

Verification:
  points → MCP algorithm → collinear set
  → recover line → compare (a₀, a₁)
```

## Analysis Tools

### Experiment Analysis

```bash
cd scripts
python analyze.py                     # Use per-experiment max_tokens as threshold
python analyze.py --min-tokens 200    # Global threshold of 200 tokens
python analyze.py --force             # Proceed despite conflicting parameters
```

Outputs:
- Text reports with statistics
- Box plots showing block distributions
- Source experiment tracking

### Attack Simulation

```bash
cd scripts
python attack_simulation.py                          # Default 10% perturbation
python attack_simulation.py --perturbation-rate 20   # 20% max perturbation
python attack_simulation.py --groups "1,2,3,4,5"     # Test different group counts
python attack_simulation.py -j 4                     # Use 4 worker processes
```

Simulates three attack types:
- **Insertion**: Add random tokens at contiguous positions
- **Deletion**: Remove contiguous token groups
- **Substitution**: Replace contiguous token groups with random tokens

Outputs:
- CSV with detailed results
- Recovery rate graphs (per-attack and combined)
- Summary reports

## Output Structure

Results are saved to timestamped directories:

```
output/
└── {model}_{dataset}_n{n}_{timestamp}/
    ├── run_config.json    # Experiment configuration
    ├── statistics.csv     # Human-readable results
    └── statistics.parquet # Efficient binary format
```

### Statistics Columns

| Column | Description |
|--------|-------------|
| `watermark_recovered` | Whether watermark was successfully verified |
| `a0`, `a1` | Original secret line coefficients |
| `recovered_a0`, `recovered_a1` | Recovered coefficients (if successful) |
| `watermark_blocks` | Intended watermark blocks (JSON) |
| `encoded_blocks` | Actually encoded blocks (JSON) |
| `decoded_blocks` | All decoded blocks (JSON) |
| `valid_blocks` | Valid blocks after filtering (JSON) |
| `matching_blocks` | Blocks matching watermark y-values (JSON) |
| `properly_encoded_tokens` | Tokens matching intended bias |
| `unique_*_count` | Unique block counts for various categories |
| `encoding_time`, `decoding_time`, `mcp_time` | Timing information |

## Important Implementation Details

### Device Consistency
**Critical**: `torch.randperm` produces different results on CPU vs CUDA. The decoder must use the same device as the encoder. This also applies to attack simulation scripts.

### X-Coordinate Generation
- First block uses token ID 0 (not the last prompt token)
- This ensures the decoder works without access to the original prompt
- Subsequent blocks use the last generated token

### Vocabulary Split
- Deterministic hash-based permutation per previous token
- LRU cache (maxsize=1000) for performance
- Split: first `green_fraction` of permuted vocab → green list

### Token Selection
- Token selection is **biased, not enforced**
- Some bits may not encode correctly due to model preferences
- `properly_encoded_tokens` tracks encoding accuracy

### Hamming Mode
- Uses systematic format: data bits followed by parity bits
- Sliding window decoder finds all valid windows
- Detection-only mode (default) has lower false positive rate than correction mode

### C-Correction Mode
- Incompatible with Hamming mode
- Generates (n choose c) variations per block for c-bit correction
- High values (>2) generate many variations and slow down MCP

## Performance Considerations

- **Memory**: Large models and long generations require significant memory
- **GPU**: CUDA support recommended for faster inference
- **Context Window**: Automatic K/V cache trimming prevents memory overflow
- **Hamming Modes**: SECDED provides better error detection than standard Hamming with minimal overhead
- **C-Correction**: Use low values (1-2) to avoid combinatorial explosion
- **Multiprocessing**: Attack simulation scales well with multiple cores

## Known Issues

### Module Import Pattern
Always import paths as a module:
```python
import src.paths as paths  # Correct - set_cache_dir() will work
from src.paths import CACHE_DIR  # Wrong - won't see updates
```

### Chat Template Warnings
Some models without proper chat templates produce AutoProcessor warnings (non-critical).

## Development Notes

- Token selection is biased, not enforced - some bits may not encode correctly
- `properly_encoded_tokens` tracks how many tokens matched intended bias direction
- C-correction with c ≥ 3 generates many variants (n choose c) - can be slow

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request with output of your tests if applicable

## Citation

If you use this implementation in your research, please cite the relevant academic papers and this repository.

## Acknowledgments

- Based on [LLM-Text Watermarking based on Lagrange Interpolation](https://arxiv.org/abs/2505.05712)
- Galois field implementation optimized by Pawel Morawiecki
- Built on HuggingFace Transformers ecosystem

## Disclaimer

This project was developed with assistance from AI coding assistants including Claude Sonnet 4, Claude Sonnet 4.5, Claude Opus 4.5, and GPT-5. All code and documentation were reviewed and approved by a human developer, with proper oversight maintained throughout the development process.
