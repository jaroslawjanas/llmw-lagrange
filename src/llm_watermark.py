#!/usr/bin/env python3
"""
LLM Watermarking Implementation based on Lagrange Interpolation
https://arxiv.org/abs/2301.10226

This script implements the Lagrange interpolation watermarking technique for LLM-generated text.
"""

import hashlib
import random
import torch
import numpy as np
import time
from itertools import combinations
from typing import List, Tuple, Dict, Optional, Union
from abc import ABC, abstractmethod
from functools import lru_cache
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from tqdm import tqdm
from src.model_formatters import format_prompt_for_model
from src.utils import load_hf_token
from src.pm_galois import GaloisField, max_collinear_points, recover_line_equation
from src.hamming import HammingCode
import src.paths as paths


@lru_cache(maxsize=1000)
def _compute_vocab_split(
    token_id: int, secret_key: str, vocab_size: int, green_list_fraction: float, device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Cached vocab split computation.

    WARNING: torch.randperm produces different results on CPU vs CUDA even with same seed.
    The decoder MUST use the same device as was used during encoding for correct results.
    This also applies to any scripts that use the decoder (e.g., attack_simulation.py).

    Args:
        token_id: Previous token ID to hash
        secret_key: Secret key for watermarking
        vocab_size: Size of vocabulary
        green_list_fraction: Fraction of tokens in green list
        device: Device for torch operations

    Returns:
        Tuple of (green_tokens, red_tokens) as tensors
    """
    hash_input = f"{token_id}-{secret_key}"
    hash_seed = int(hashlib.sha256(hash_input.encode()).hexdigest(), 16) % (2**32)

    rng = torch.Generator(device=device)
    rng.manual_seed(hash_seed)

    perm = torch.randperm(vocab_size, generator=rng, requires_grad=False, device=device)
    split = int(vocab_size * green_list_fraction)

    return perm[:split], perm[split:]


class LLMWatermarkerBase(ABC):
    """
    Abstract base class for LLM watermarking encoder and decoder.
    Contains shared functionality for both encoding and decoding operations.
    """
    
    def __init__(
        self,
        model_name: str,
        secret_key: str,
        n: int,
        gf: object,
        device: str,
        green_list_fraction: float = 0.5,
        seed: int = 4242,
        cache_dir: str = paths.CACHE_DIR,
        verbose: bool = False,
    ):
        """
        Initialize the base watermarker with shared parameters.
        
        Args:
            model_name: HuggingFace model identifier
            secret_key: Secret key for watermarking
            n: Field size parameter (GF(2^n))
            gf: Galois field instance GF(2^n)
            green_list_fraction: Fraction of tokens to include in green list (default: 0.5)
            seed: Random seed for reproducibility
            cache_dir: Directory to cache models
            device: Device to run on ('cuda', 'cpu', or None for auto-detection)
            verbose: Whether to show detailed output
        """
        self.model_name = model_name
        self.secret_key = secret_key
        self.n = n
        self.gf = gf
        self.green_list_fraction = green_list_fraction
        self.seed = seed
        self.cache_dir = cache_dir
        self.device = device
        self.verbose = verbose
        
        # Load tokenizer (shared by both encoder and decoder)
        self._load_tokenizer()
        self.vocab_size = len(self.tokenizer)
        
    def _load_tokenizer(self):
        """Load the tokenizer."""
        if self.verbose:
            print(f"Loading tokenizer for: {self.model_name}")
        
        # Get HuggingFace token if available
        token = load_hf_token()
        
        # Configure tokenizer options
        tokenizer_kwargs = {
            "cache_dir": paths.MODELS_CACHE_DIR,  # Use models subdirectory
        }
        if token:
            tokenizer_kwargs["token"] = token
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            **tokenizer_kwargs
        )
        
        # Ensure the tokenizer has a padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _hash_to_gf_element(self, token_id: int, secret_key: str) -> object:
        """
        Hash a token ID and secret key to produce a GF(2^n) element.
        
        Args:
            token_id: Token ID to hash
            secret_key: Secret key for watermarking
            
        Returns:
            Element in GF(2^n)
        """
        # Create hash input by concatenating token ID and secret key
        hash_input = f"{token_id}-{secret_key}"
        hash_object = hashlib.sha256(hash_input.encode())
        hash_bytes = hash_object.digest()
        
        # Take first n bits from the hash
        # Convert bytes to integer, then take modulo 2^n
        hash_int = int.from_bytes(hash_bytes[:4], byteorder='big')  # Use first 4 bytes
        hash_value = hash_int % (2 ** self.n)
        
        # Convert to GF(2^n) element
        return self.gf(hash_value)
    
    def _gf_to_binary(self, gf_element: object) -> List[int]:
        """
        Convert a GF(2^n) element to its n-bit binary representation.
        
        Args:
            gf_element: Element in GF(2^n)
            
        Returns:
            List of n bits (0s and 1s)
        """
        # Get the integer representation of the field element
        int_value = int(gf_element)
        
        # Convert to binary and pad to n bits
        binary_str = format(int_value, f'0{self.n}b')
        
        # Convert to list of integers
        return [int(bit) for bit in binary_str]
    
    def _binary_to_gf(self, binary_bits: List[int]) -> object:
        """
        Convert a binary sequence to a GF(2^n) element.
        
        Args:
            binary_bits: List of n bits (0s and 1s)
            
        Returns:
            Element in GF(2^n)
        """
        if len(binary_bits) != self.n:
            raise ValueError(f"Binary sequence must have exactly {self.n} bits, got {len(binary_bits)}")
        
        # Convert binary list to integer
        binary_str = ''.join(str(bit) for bit in binary_bits)
        int_value = int(binary_str, 2)
        
        # Convert to GF(2^n) element
        return self.gf(int_value)
    
    def _get_red_green_tokens(self, token_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate red and green token lists based on the hash of a token and secret key.
        This creates a deterministic division of the vocabulary into "green" and "red" tokens.
        Uses LRU cache (maxsize=1000) to avoid recomputing for repeated token_ids.

        Args:
            token_id: Single token ID to hash

        Returns:
            Tuple of (green_tokens, red_tokens) as tensors
        """
        return _compute_vocab_split(
            token_id, self.secret_key, self.vocab_size, self.green_list_fraction, self.device
        )


class LLMWatermarkEncoder(LLMWatermarkerBase):
    def __init__(
        self,
        model_name: str,
        secret_key: str,
        line_fnc: callable,
        n: int,
        gf: object,
        device: str,
        green_list_fraction: float = 0.5,
        bias: float = 6.0,
        seed: int = 42,
        cache_dir: str = paths.CACHE_DIR,
        context_window: int = 1500,
        temperature: float = 0.0,
        hash_window: int = 1,
        hamming_mode: str = "none",
        verbose: bool = False
    ):
        """
        Initialize the watermarker with the specified model and parameters.

        Args:
            model_name: HuggingFace model identifier
            secret_key: Secret key for watermarking
            line_fnc: Function for Lagrange interpolation f(x) = a0 + a1*x
            n: Field size parameter (GF(2^n))
            gf: Galois field instance GF(2^n)
            green_list_fraction: Fraction of tokens to include in green list (default: 0.5)
            bias: Logit bias to apply to green tokens (default: 6.0)
            seed: Random seed for reproducibility
            cache_dir: Directory to cache models
            device: Device to run model on ('cuda', 'cpu')
            context_window: Maximum number of tokens to use as context for generation (default: 1024)
            temperature: Sampling temperature (default: 0.0 = greedy sampling, higher = more random)
            hash_window: Number of previous tokens to hash together (default: 1)
            hamming_mode: Hamming code mode ("none", "standard", "secded")
        """
        # Initialize base class
        super().__init__(model_name, secret_key, n, gf, device, green_list_fraction, seed, cache_dir, verbose)

        # Encoder-specific parameters
        self.line_fnc = line_fnc
        self.bias = bias
        self.context_window = context_window
        self.temperature = temperature
        self.hash_window = hash_window

        # Hamming code setup
        self.hamming_mode = hamming_mode
        if hamming_mode != "none":
            self.hamming = HammingCode(n, secded=(hamming_mode == "secded"))
        else:
            self.hamming = None

        # Load model (tokenizer already loaded by base class)
        self._load_model()

    @property
    def tokens_per_block(self) -> int:
        """Tokens per watermark block: n for standard, n + parity_bits for Hamming."""
        if self.hamming:
            return self.n + self.hamming.parity_bit_count
        return self.n
        
    def _load_model(self):
        """Load the model."""
        if self.verbose:
            print(f"Loading model: {self.model_name}")
        
        # Get HuggingFace token if available
        token = load_hf_token()
        
        # Configure model loading options
        model_kwargs = {
            "cache_dir": paths.MODELS_CACHE_DIR,  # Use models subdirectory
            "use_cache": True,  # Ensure K/V caching is enabled
        }
        if token:
            model_kwargs["token"] = token
        
        # Load model with appropriate settings for the device
        if self.device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                torch_dtype=torch.float16,  # Use float16 for memory efficiency
                device_map="auto",
                **model_kwargs
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            self.model.to(self.device)
            
    def _trim_past_key_values(self, past_key_values: Tuple, trim_amount: int) -> Tuple:
        """
        Trim the K/V cache by removing the oldest entries.
        
        This method is essential for K/V caching because:
        1. Memory Management: K/V caches grow linearly with sequence length, potentially causing OOM errors
        2. Context Window Limits: Models have maximum context windows (e.g., 2048, 4096 tokens)
        3. Performance: Without trimming, the cache would grow indefinitely during long text generation
        4. Correctness: Prevents exceeding model's positional encoding limits
        
        The method removes the oldest tokens from the beginning of the cache while preserving
        the most recent context, which is typically most relevant for generation.
        
        Args:
            past_key_values: Tuple of past key values from the model
            trim_amount: Number of positions to remove from the beginning
            
        Returns:
            Trimmed past_key_values tuple
        """
        trimmed_past = []
        
        # Iterate through each layer's key-value pairs
        for layer_past in past_key_values:
            # Each layer_past is a tuple of (key, value) tensors
            # Shape: (batch_size, num_heads, sequence_length, head_dim)
            key, value = layer_past
            
            # Trim the sequence dimension (dimension 2) by removing oldest tokens
            trimmed_key = key[:, :, trim_amount:, :]
            trimmed_value = value[:, :, trim_amount:, :]
            
            # Append the trimmed tuple for this layer
            trimmed_past.append((trimmed_key, trimmed_value))
        
        return tuple(trimmed_past)
        
    def _modify_logits(self, logits: torch.Tensor, token_id: int, bias_type: str) -> torch.Tensor:
        """
        Modify logits by adding bias to either green or red tokens.
        This is the core watermarking function that biases the model's predictions
        toward selecting tokens from the specified list determined by previous token and secret key.
        
        Args:
            logits: Original logits from the model (prediction scores for each token)
            token_id: Previous token ID to use for hashing
            secret_key: Secret key for watermarking
            bias_type: Type of tokens to bias ("green" or "red")
            
        Returns:
            Modified logits tensor with bias added to the specified tokens
        """
        # Ensure logits are in the right shape (batch_size, seq_len, vocab_size)
        # The tensor might be (1, 1, vocab_size) or just (1, vocab_size)
        if len(logits.shape) == 3:
            # Get the last token's logits and flatten to 1D
            # Example: if logits is tensor([[[1.2, 0.8, -0.5, ...]]])
            # This becomes tensor([1.2, 0.8, -0.5, ...])
            logits = logits[0, -1, :]
        elif len(logits.shape) == 2:
            # Already (batch_size, vocab_size), get first batch
            # Example: if logits is tensor([[1.2, 0.8, -0.5, ...]])
            # This becomes tensor([1.2, 0.8, -0.5, ...])
            logits = logits[0, :]
        
        # Get vocabulary size from logits
        # Example: if logits tensor has 50,000 elements, vocab_size will be 50,000
        vocab_size = logits.shape[-1]
        
        # Get green and red tokens as tensors using the deterministic hash of previous token and secret key
        # Example: based on token_id 101 and secret key, this might return
        # green_tokens as tensor([42, 900, 5, ...]) and red_tokens (which we don't use here)
        green_tokens, red_tokens = self._get_red_green_tokens(token_id)
        
        # Move tensors to the same device as logits
        green_tokens = green_tokens.to(logits.device)
        red_tokens = red_tokens.to(logits.device)
        
        # Clone logits for modification to avoid affecting the original tensor
        # This creates a new tensor with the same values that we can safely modify
        modified_logits = logits.clone()
        
        if bias_type == "green":
            # Filter green tokens to ensure they're within vocabulary bounds using tensor operations
            # Example: if vocab_size is 50,000 and green_tokens contains a value 60,000,
            # we create a boolean mask tensor([True, True, True, False, ...]) for values < 50,000
            mask = green_tokens < vocab_size
            # Apply the mask to get only valid green tokens
            # Example: valid_green_tokens becomes tensor([42, 900, 5, ...]) without any out-of-bounds tokens
            valid_tokens_to_bias = green_tokens[mask]
        elif bias_type == "red":
            # Filter red tokens to ensure they're within vocabulary bounds
            mask = red_tokens < vocab_size
            valid_tokens_to_bias = red_tokens[mask]
        else:
            raise ValueError("bias_type must be 'green' or 'red'")
        
        # Vectorized bias application - apply bias to all valid tokens at once
        # This means tokens in the specified list will have higher probability of being selected
        if valid_tokens_to_bias.numel() > 0:  # Only proceed if we have valid tokens
            # Example: if self.bias is 6.0, this adds 6.0 to the logits for all specified tokens
            # For the token indices in valid_tokens_to_bias
            modified_logits[valid_tokens_to_bias] += self.bias
        
        # Reshape back to original format for compatibility with the model's expectations
        # Example: from tensor([1.2, 0.8, -0.5, ...]) back to tensor([[[1.2, 0.8, -0.5, ...]]])
        return modified_logits.unsqueeze(0).unsqueeze(0)
        
    def generate_text(
        self, 
        prompt: str, 
        max_new_tokens: int = 100, 
        verbose: bool = True
    ) -> Tuple[str, str, List[int], str, Dict[str, int], List[Dict], List[Dict]]:
        """
        Generate text with Lagrange interpolation watermarking.

        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            verbose: Whether to show progress bar

        Returns:
            Tuple of (full_text, generated_text, generated_ids, formatted_prompt, statistics, watermark_blocks, encoded_blocks)
            - watermark_blocks: List of intended watermark blocks {x, y, y_bits, p_bits}
            - encoded_blocks: List of actually encoded blocks {x, y, y_bits, p_bits}
        """

        # Reset counters
        self.green_tokens_selected = 0
        self.red_tokens_selected = 0
        self.blocks_encoded = 0
        self.properly_encoded_tokens = 0

        # Initialize tracking for watermark blocks
        watermark_blocks = []
        encoded_blocks = []
        
        # Format the prompt for the specific model
        formatted_prompt = format_prompt_for_model(prompt, self.model_name, self.tokenizer, verbose=self.verbose)
        
        # Tokenize the formatted prompt
        input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)

        # Store generated ids
        all_ids = input_ids.clone()[0].tolist()
        
        # Initialize K/V cache
        past_key_values = None
        cache_position = 0  # Track position in the cache
        
        # Calculate number of complete blocks we can generate
        if max_new_tokens % self.tokens_per_block != 0:
            block_desc = f"{self.n} data" + (f" + {self.hamming.parity_bit_count} parity" if self.hamming else "")
            raise ValueError(
                f"max_new_tokens ({max_new_tokens}) must be divisible by "
                f"tokens_per_block ({self.tokens_per_block} = {block_desc}) to ensure complete blocks."
            )
        num_blocks = max_new_tokens // self.tokens_per_block
        if num_blocks == 0:
            raise ValueError(f"max_new_tokens ({max_new_tokens}) must be at least tokens_per_block ({self.tokens_per_block}) to generate at least one block.")
        
        # Setup progress tracking
        progress_bar = tqdm(range(max_new_tokens))
        tokens_generated = 0
        
        # Generate tokens block by block
        for block_idx in range(num_blocks):
            # At the start of each block, determine the token to use for x-coordinate computation
            if len(all_ids) > len(input_ids[0]):
                # Use the last generated token for subsequent blocks
                previous_token_id = all_ids[-1]
            else:
                # Use 0 for the very first block instead of last prompt token
                # This is because when running detection we do not have access to the prompt
                previous_token_id = 0
            
            # Compute x-coordinate for this block
            x = self._hash_to_gf_element(previous_token_id, self.secret_key)
            
            # Compute y = f(x) using the line function
            y = self.line_fnc(x)

            # Convert y to binary representation
            y_bits = self._gf_to_binary(y)

            # Apply Hamming encoding if enabled
            if self.hamming:
                block_bits, p_bits = self.hamming.encode(y_bits)
            else:
                block_bits = y_bits
                p_bits = []

            # Track encoded bits for this block
            encoded_bits = []

            # Generate tokens for this block (one for each bit)
            for bit_idx in range(self.tokens_per_block):
                # Prepare input for the model
                if past_key_values is None:
                    # First generation - use the full prompt
                    # Only use the last context_window tokens if needed
                    if len(all_ids) > self.context_window:
                        model_input_ids = torch.tensor([all_ids[-self.context_window:]], device=self.device)
                        cache_position = 0  # Reset cache position if we truncate
                    else:
                        model_input_ids = torch.tensor([all_ids], device=self.device)
                else:
                    # Subsequent generations - only use the last generated token
                    model_input_ids = torch.tensor([[all_ids[-1]]], device=self.device)
                    
                    # Check if we need to trim the cache due to context window limits
                    if cache_position >= self.context_window:
                        # Trim the K/V cache to stay within context window
                        # Keep only the most recent tokens
                        trim_amount = cache_position - self.context_window + 1
                        past_key_values = self._trim_past_key_values(past_key_values, trim_amount)
                        cache_position = self.context_window - 1
                
                # Get logits from the model with K/V caching
                with torch.no_grad():
                    outputs = self.model(
                        model_input_ids, 
                        past_key_values=past_key_values,
                        use_cache=True  # Enable K/V caching
                    )
                    logits = outputs.logits[:, -1:, :]  # Get logits of last token
                    past_key_values = outputs.past_key_values  # Update cache
                    cache_position += model_input_ids.shape[1]  # Update position
                
                # Determine bias type based on current bit (from block_bits which includes Hamming parity if enabled)
                current_bit = block_bits[bit_idx]
                bias_type = "green" if current_bit == 1 else "red"
                
                # For vocabulary splitting - use token_id = 0 for very first token, otherwise use previous generated token
                if tokens_generated == 0:
                    # Very first token being generated
                    current_previous_token = 0
                else:
                    # Use the previous generated token (never use prompt tokens)
                    current_previous_token = all_ids[-1]
                
                # Modify logits with watermark using the current previous token and secret key
                modified_logits = self._modify_logits(logits, current_previous_token, bias_type)
                
                # Apply temperature if set
                if self.temperature > 0:
                    # Scale logits by temperature
                    modified_logits = modified_logits / self.temperature
                    
                # Get probabilities through softmax
                probs = torch.nn.functional.softmax(modified_logits, dim=-1)
                
                # Token sampling based on temperature
                if self.temperature == 0 or self.temperature < 1e-6:
                    # Greedy sampling (select token with highest probability)
                    next_token_id = torch.argmax(probs, dim=-1).item()
                else:
                    # Sample from the distribution
                    next_token_id = torch.multinomial(probs.squeeze(), 1).item()
                
                # Track whether the selected token was from the green or red list for watermark statistics
                # Get the current division of tokens into green and red based on previous token and secret key
                green_tokens, red_tokens = self._get_red_green_tokens(current_previous_token)
                
                # Check vocabulary bounds for safety
                vocab_size = len(self.tokenizer)
                if next_token_id < vocab_size:  # Make sure token is in vocabulary range
                    # Create a tensor from the next token ID to enable vectorized comparison
                    next_token_tensor = torch.tensor(next_token_id, device=self.device)
                    
                    # Use tensor operations to efficiently check if the token is in the green list
                    is_green = (green_tokens == next_token_tensor).any().item()
                    encoded_bit = 1 if is_green else 0
                    
                    if is_green:
                        encoded_bits.append(1)
                        self.green_tokens_selected += 1
                    else:
                        encoded_bits.append(0)
                        self.red_tokens_selected += 1
                    
                    # Check if token was properly encoded (intended bit matches actual encoded bit)
                    if current_bit == encoded_bit:
                        self.properly_encoded_tokens += 1

                # Add the new token to generated ids
                all_ids.append(next_token_id)
                tokens_generated += 1
                
                # Update progress bar with stats
                progress_bar.set_description(f"Encoding Block {block_idx+1}/{num_blocks}, Bit {bit_idx+1}/{self.tokens_per_block}, Green: {self.green_tokens_selected}, Red: {self.red_tokens_selected}")
                progress_bar.update(1)
                
                # Check if we've reached an EOS token
                if next_token_id == self.tokenizer.eos_token_id:
                    break
            
            # Build watermark block (intended)
            watermark_block = {
                'x': int(x),
                'y': int(y),
                'y_bits': y_bits.copy(),
                'p_bits': list(p_bits) if self.hamming else []
            }
            watermark_blocks.append(watermark_block)

            # Build encoded block (actual)
            encoded_y_bits = encoded_bits[:self.n]
            encoded_p_bits = encoded_bits[self.n:] if self.hamming else []
            encoded_y = self._binary_to_gf(encoded_y_bits)
            encoded_block = {
                'x': int(x),
                'y': int(encoded_y),
                'y_bits': encoded_y_bits,
                'p_bits': encoded_p_bits
            }
            encoded_blocks.append(encoded_block)

            self.blocks_encoded += 1
            
            # Check if we hit EOS in the middle of a block
            if all_ids[-1] == self.tokenizer.eos_token_id:
                break
        
        progress_bar.close()

        # Separate prompt from generated content using existing input_ids
        prompt_length = len(input_ids[0])  # input_ids is a tensor with batch dimension
        
        # Split the generated_ids into prompt and generated portions
        if len(all_ids) > prompt_length:
            generated_ids = all_ids[prompt_length:]
            assert len(generated_ids) == tokens_generated
        else:
            generated_ids = []  # No new tokens were generated
        
        # Decode both full text and generated-only text
        full_text = self.tokenizer.decode(all_ids, skip_special_tokens=True)
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True) if generated_ids else ""
        
        # Create statistics dictionary
        statistics = {
            'green_tokens': self.green_tokens_selected,
            'red_tokens': self.red_tokens_selected,
            'blocks_encoded': self.blocks_encoded,
            'total_tokens_generated': tokens_generated,
            'green_ratio': self.green_tokens_selected / (self.green_tokens_selected + self.red_tokens_selected + 1e-10),
            'properly_encoded_tokens': self.properly_encoded_tokens
        }

        return full_text, generated_text, generated_ids, formatted_prompt, statistics, watermark_blocks, encoded_blocks


class LLMWatermarkDecoder(LLMWatermarkerBase):
    """
    Decoder for extracting watermark information from LLM-generated text.
    Reverses the encoding process to extract x-coordinates and y-bit sequences.
    """

    def __init__(
        self,
        model_name: str,
        secret_key: str,
        n: int,
        gf: object,
        device: str,
        green_list_fraction: float = 0.5,
        seed: int = 4242,
        cache_dir: str = paths.CACHE_DIR,
        verbose: bool = False,
        hamming_mode: str = "none",
        correct: bool = False,
        c_correction: int = 0,
    ):
        """
        Initialize the decoder with the same parameters used for encoding.

        Args:
            model_name: HuggingFace model identifier (same as used for encoding)
            secret_key: Secret key for watermarking (same as used for encoding)
            n: Field size parameter (GF(2^n)) (same as used for encoding)
            gf: Galois field instance GF(2^n) (same as used for encoding)
            device: Device to run on ('cuda' or 'cpu')
            green_list_fraction: Fraction of tokens in green list (same as used for encoding)
            seed: Random seed for reproducibility (same as used for encoding)
            cache_dir: Directory to cache models
            verbose: Whether to show detailed output
            hamming_mode: Hamming code mode ("none", "standard", "secded")
            correct: Whether to enable Hamming error correction (False = detection-only)
            c_correction: Max Hamming distance for variation generation (0 = disabled)
        """
        # Initialize base class (loads tokenizer)
        super().__init__(model_name, secret_key, n, gf, device, green_list_fraction, seed, cache_dir, verbose)

        # Hamming code setup
        self.hamming_mode = hamming_mode
        self.correct = correct
        self.c_correction = c_correction
        if hamming_mode != "none":
            self.hamming = HammingCode(n, secded=(hamming_mode == "secded"))
        else:
            self.hamming = None

    @property
    def tokens_per_block(self) -> int:
        """Tokens per watermark block: n for standard, n + parity_bits for Hamming."""
        if self.hamming:
            return self.n + self.hamming.parity_bit_count
        return self.n
        
    def decode_text(self, generated_text: str = None, generated_ids: List[int] = None) -> Tuple[List[Dict], List[Dict], int]:
        """
        Decode watermark information from generated text or pre-tokenized IDs.

        Args:
            generated_text: The generated text to decode (prompt already removed)
            generated_ids: Pre-tokenized IDs to decode directly (bypasses tokenization)

        Returns:
            Tuple of (all_blocks, valid_blocks, token_count) where each block contains:
            - 'x': x-coordinate (GF element as integer)
            - 'y': y-value as GF element converted to integer
            - 'y_bits': Data bits (flat list)
            - 'p_bits': Parity bits (empty list for non-Hamming)
        """
        # Get token IDs from text or use provided IDs
        if generated_ids is not None:
            token_ids = generated_ids
        elif generated_text is not None:
            token_ids = self.tokenizer.encode(generated_text, add_special_tokens=False)
        else:
            raise ValueError("Either generated_text or generated_ids must be provided")

        # Dispatch to appropriate decoder
        if self.hamming:
            return self._decode_sliding_blocks(token_ids)
        elif self.c_correction > 0:
            # Decode fixed blocks, then apply c-correction variations
            all_blocks, _, token_count = self._decode_fixed_blocks(token_ids)
            expanded_blocks = self._apply_c_correction(all_blocks)
            return expanded_blocks, expanded_blocks, token_count
        else:
            return self._decode_fixed_blocks(token_ids)

    def _decode_tokens_to_bits(self, token_ids: List[int]) -> List[int]:
        """
        Convert all tokens to bits based on green/red classification.

        Args:
            token_ids: List of token IDs to classify

        Returns:
            List of bits (1 for green, 0 for red)
        """
        all_bits = []
        for i, token_id in enumerate(token_ids):
            # Previous token for vocab split (0 for first token)
            prev_token = 0 if i == 0 else token_ids[i - 1]
            green_tokens, _ = self._get_red_green_tokens(prev_token)
            is_green = (green_tokens == token_id).any().item()
            all_bits.append(1 if is_green else 0)
        return all_bits

    def _decode_fixed_blocks(self, token_ids: List[int]) -> Tuple[List[Dict], List[Dict], int]:
        """
        Decode using fixed-boundary blocks (non-Hamming mode).

        Args:
            token_ids: List of token IDs to decode

        Returns:
            Tuple of (all_blocks, valid_blocks, token_count)
            For non-Hamming, all_blocks and valid_blocks are the same list.
        """
        num_blocks = len(token_ids) // self.n

        if num_blocks <= 1:
            return [], [], 0

        blocks = []
        progress_bar = tqdm(range(num_blocks), desc="Decoding Blocks", disable=not self.verbose)

        for block_idx in range(num_blocks):
            block_start = block_idx * self.n

            # X-coordinate from token before this block (0 for first block)
            prev_token = 0 if block_start == 0 else token_ids[block_start - 1]
            x = self._hash_to_gf_element(prev_token, self.secret_key)

            # Classify each token in block as green (1) or red (0)
            y_bits = []
            for i in range(self.n):
                pos = block_start + i
                token_id = token_ids[pos]
                vocab_split_token = 0 if pos == 0 else token_ids[pos - 1]
                green_tokens, _ = self._get_red_green_tokens(vocab_split_token)
                is_green = (green_tokens == token_id).any().item()
                y_bits.append(1 if is_green else 0)

            y_gf = self._binary_to_gf(y_bits)
            blocks.append({
                'x': int(x),
                'y': int(y_gf),
                'y_bits': y_bits,
                'p_bits': []
            })
            progress_bar.update(1)

        progress_bar.close()
        return blocks, blocks, len(token_ids)

    def _decode_sliding_blocks(self, token_ids: List[int]) -> Tuple[List[Dict], List[Dict], int]:
        """
        Decode using sliding window with Hamming validity check.

        Slides a window of size tokens_per_block over the bit stream,
        checking each window for Hamming validity. Returns all blocks and
        valid blocks separately.

        Args:
            token_ids: List of token IDs to decode

        Returns:
            Tuple of (all_blocks, valid_blocks, token_count)
            - all_blocks: Every sliding window result
            - valid_blocks: Only Hamming-valid windows
        """
        if not self.hamming:
            raise ValueError("Sliding window decode requires Hamming mode")

        decoded_tokens_length = len(token_ids)

        # Step 1: Convert all tokens to bits
        if self.verbose:
            print(f"Converting {len(token_ids)} tokens to bits...")
        all_bits = self._decode_tokens_to_bits(token_ids)

        # Step 2: Slide window and collect all blocks
        window_size = self.tokens_per_block
        all_blocks = []
        valid_blocks = []
        valid_count = 0
        invalid_count = 0

        if self.verbose:
            print(f"Sliding window of size {window_size} over {len(all_bits)} bits...")

        num_windows = len(all_bits) - window_size + 1
        progress_bar = tqdm(range(num_windows), desc="Sliding Window Decode", disable=not self.verbose)

        for start in progress_bar:
            window_bits = all_bits[start:start + window_size]

            # Systematic format: parity bits are at the end
            p_bits = window_bits[self.n:]

            # Decode and check validity using Hamming
            data_bits, syndrome, is_valid = self.hamming.decode(window_bits, correct=self.correct)

            # Compute x from token before this window
            prev_token = 0 if start == 0 else token_ids[start - 1]
            x = self._hash_to_gf_element(prev_token, self.secret_key)
            y_gf = self._binary_to_gf(data_bits)

            block = {
                'x': int(x),
                'y': int(y_gf),
                'y_bits': data_bits,
                'p_bits': p_bits
            }

            all_blocks.append(block)

            if is_valid:
                valid_count += 1
                valid_blocks.append(block)
            else:
                invalid_count += 1

            progress_bar.set_description(f"Sliding Window: {valid_count} valid, {invalid_count} invalid")

        progress_bar.close()

        if self.verbose:
            print(f"Found {valid_count} valid windows, {invalid_count} invalid windows")

        return all_blocks, valid_blocks, decoded_tokens_length

    def _generate_bit_variations(self, bits: List[int], max_distance: int) -> List[List[int]]:
        """
        Generate all bit-flip variations of a bit sequence up to a given Hamming distance.

        Args:
            bits: Original bit sequence
            max_distance: Maximum number of bits to flip (e.g., 1 or 2)

        Returns:
            List of all variations including the original
        """
        n = len(bits)
        variations = [bits.copy()]  # Include original

        # Generate all combinations of positions to flip for each distance
        for distance in range(1, max_distance + 1):
            for positions in combinations(range(n), distance):
                variant = bits.copy()
                for pos in positions:
                    variant[pos] ^= 1  # Flip the bit
                variations.append(variant)

        return variations

    def _apply_c_correction(self, blocks: List[Dict]) -> List[Dict]:
        """
        Apply c-correction by generating all bit-flip variations for each block.

        Args:
            blocks: List of decoded blocks from _decode_fixed_blocks

        Returns:
            List: [orig_1, orig_2, ..., orig_k, corr_1_1, corr_1_2, ..., corr_2_1, ...]
        """
        all_blocks = []

        # First: add all original blocks
        for block in blocks:
            all_blocks.append({
                'x': block['x'],
                'y': block['y'],
                'y_bits': block['y_bits'].copy(),
                'p_bits': []
            })

        # Second: add all corrections
        for block in blocks:
            variations = self._generate_bit_variations(block['y_bits'], self.c_correction)

            for var_bits in variations[1:]:  # Skip index 0 (original)
                y_gf = self._binary_to_gf(var_bits)
                all_blocks.append({
                    'x': block['x'],
                    'y': int(y_gf),
                    'y_bits': var_bits,
                    'p_bits': []
                })

        if self.verbose:
            print(f"Generated {len(all_blocks)} blocks from {len(blocks)} original blocks (c={self.c_correction})")

        return all_blocks


class MCPSolver:
    """
    Maximum Collinear Points solver for watermark verification.
    Implements the hashing-based algorithm with O(N²) complexity.

    Contrary to the rest of the project this class uses Pawel's GF implementation
    as the original galois package proved to be too slow. Rather to re-implement the entire galois
    setup we simply modified this function to use a faster (Pawel's) implementation.
    """
    
    def __init__(self, gf: object, n: int, verbose: bool = False):
        """
        Initialize the MCP solver.
        
        Args:
            gf: Galois field instance GF(2^n)
            n: Field size parameter
            verbose: Whether to show detailed output
        """
        self.gf = gf
        self.n = n
        self.verbose = verbose
    
    def solve_mcp(self, points: List[Tuple[int, int]]) -> Tuple[int, int, List[Tuple[int, int]]]:
        """
        Solve the Maximum Collinear Points problem using the new GaloisField framework.
        
        Args:
            points: List of (x, y) tuples as integers
            
        Returns:
            Tuple of (max_count, best_slope, collinear_points)
            - max_count: Maximum number of collinear points found
            - best_slope: Slope of the line with most points (as integer)
            - collinear_points: List of points on the best line as integers
        """
        if len(points) < 2:
            return 0, None, []

        # Use the new framework's max_collinear_points function
        max_count, best_slope, collinear_points = max_collinear_points(points, self.gf)
        
        if self.verbose:
            print(f"Found maximum {max_count} collinear points with slope {best_slope}")
        
        return max_count, best_slope, collinear_points
    
    def recover_line_equation(self, collinear_points: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        Recover the line equation f(x) = a₀ + a₁x from collinear points using the new GaloisField framework.
        
        Args:
            collinear_points: List of collinear points as integer tuples
            
        Returns:
            Tuple of (a₀, a₁) as integers
        """
        if len(collinear_points) < 2:
            raise ValueError("Need at least 2 points to determine a line")

        # Use the new framework's recover_line_equation function
        a0, a1 = recover_line_equation(collinear_points, self.gf)
        
        return a0, a1
    
    def verify_watermark(self, decoded_blocks: List[Dict], original_a0: object, original_a1: object, watermark_blocks: List[Dict] = None) -> Dict:
        """
        Complete watermark verification pipeline.

        Args:
            decoded_blocks: List of decoded blocks (typically valid_blocks from decoder)
            original_a0: Original a0 coefficient (GF element)
            original_a1: Original a1 coefficient (GF element)
            watermark_blocks: Optional list of original watermark blocks for matching

        Returns:
            Dictionary containing:
            - 'is_valid': Boolean indicating if watermark is valid
            - 'recovered_a0': Recovered a0 coefficient
            - 'recovered_a1': Recovered a1 coefficient
            - 'max_collinear_count': Number of points on the best line
            - 'total_points': Total number of points analyzed
            - 'matching_blocks': List of decoded blocks whose y matches any watermark y
        """
        if not decoded_blocks:
            return {
                'is_valid': False,
                'recovered_a0': None,
                'recovered_a1': None,
                'max_collinear_count': 0,
                'total_points': 0,
                'matching_blocks': []
            }

        # Find matching blocks: decoded blocks whose y matches any watermark block's y
        matching_blocks = []
        if watermark_blocks and decoded_blocks:
            watermark_y_values = {wb['y'] for wb in watermark_blocks}
            for block in decoded_blocks:
                if block['y'] in watermark_y_values:
                    matching_blocks.append(block)

        # Extract (x, y) points - y is now a single int, not a list
        points = []
        for block in decoded_blocks:
            points.append((block['x'], block['y']))

        if self.verbose:
            print(f"Verifying watermark with {len(points)} points...")

        # Solve MCP problem
        max_count, best_slope, collinear_points = self.solve_mcp(points)

        if max_count < 2:
            return {
                'is_valid': False,
                'recovered_a0': None,
                'recovered_a1': None,
                'max_collinear_count': max_count,
                'total_points': len(points),
                'matching_blocks': []
            }

        # Recover line equation
        try:
            recovered_a0, recovered_a1 = self.recover_line_equation(collinear_points)
        except ValueError as e:
            if self.verbose:
                print(f"Failed to recover line equation: {e}")
            return {
                'is_valid': False,
                'recovered_a0': None,
                'recovered_a1': None,
                'max_collinear_count': max_count,
                'total_points': len(points),
                'matching_blocks': []
            }

        # Check if recovered coefficients match original
        is_valid = (recovered_a0 == original_a0) and (recovered_a1 == original_a1)

        return {
            'is_valid': is_valid,
            'recovered_a0': recovered_a0,
            'recovered_a1': recovered_a1,
            'max_collinear_count': max_count,
            'total_points': len(points),
            'matching_blocks': matching_blocks
        }
