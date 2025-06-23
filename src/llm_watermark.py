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
from typing import List, Tuple, Dict, Optional, Union
from abc import ABC, abstractmethod
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from tqdm import tqdm
from src.model_formatters import format_prompt_for_model
from src.utils import load_hf_token
import src.paths as paths


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
        green_list_fraction: float = 0.5,
        seed: int = 4242,
        cache_dir: str = paths.CACHE_DIR,
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
        """
        self.model_name = model_name
        self.secret_key = secret_key
        self.n = n
        self.gf = gf
        self.green_list_fraction = green_list_fraction
        self.seed = seed
        self.cache_dir = cache_dir
        
        # Load tokenizer (shared by both encoder and decoder)
        self._load_tokenizer()
        
    def _load_tokenizer(self):
        """Load the tokenizer."""
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
    
    def _get_red_green_tokens(self, token_id: int, secret_key: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate red and green token lists based on the hash of a token and secret key.
        This creates a deterministic division of the vocabulary into "green" and "red" tokens.
        
        Args:
            token_id: Single token ID to hash
            secret_key: Secret key for watermarking
            
        Returns:
            Tuple of (green_tokens, red_tokens) as tensors
        """
        # Create a hash from the token ID and secret key
        hash_input = f"{token_id}-{secret_key}"
        hash_object = hashlib.sha256(hash_input.encode())
        hash_hex = hash_object.hexdigest()
        
        # Convert the hexadecimal hash to a 32-bit integer for use as a random seed
        # This ensures deterministic outcomes for the same input token and secret key
        hash_seed = int(hash_hex, 16) % (2**32)
        
        # Get the total number of tokens in the model's vocabulary
        vocab_size = len(self.tokenizer)
        
        # Create a tensor containing all possible token indices (0 to vocab_size-1)
        # Example: if vocab_size is 3000, all_tokens becomes tensor([0, 1, 2, ..., 2999])
        all_tokens = torch.arange(vocab_size)
        
        # PyTorch doesn't have a direct seeded shuffle operation, so we implement
        # a deterministic shuffle using random values and sorting
        
        # Set PyTorch's random generator to use our hash-derived seed
        # This makes the shuffling deterministic based on the input tokens
        torch.manual_seed(hash_seed)
        
        # Create random values and sort them to generate a permutation of indices
        # Example: random_values might be tensor([0.12, 0.95, 0.33, ...])
        random_values = torch.rand(vocab_size)
        # Sort these values to get a permutation tensor
        # Example: permutation might be tensor([0, 2, 1, ...]) if 0.12 is smallest, 0.33 is next, etc.
        _, permutation = torch.sort(random_values)
        # Use this permutation to shuffle the token indices
        # This creates a deterministic random ordering of all vocabulary tokens
        shuffled_tokens = all_tokens[permutation]
        
        # Reset the random seed back to the original seed of the LLMWatermarker
        # This prevents the temporary seed from affecting other random operations
        torch.manual_seed(self.seed)
        
        # Split the shuffled tokens into "green" and "red" lists
        # Green tokens will get a positive bias during generation
        split_point = int(vocab_size * self.green_list_fraction)
        # Example: if green_list_fraction is 0.5 and vocab_size is 3000,
        # the first 1500 tokens in shuffled_tokens become green tokens
        green_tokens = shuffled_tokens[:split_point]  # Keep as tensor for efficiency
        # The remaining tokens become red tokens
        red_tokens = shuffled_tokens[split_point:]    # Keep as tensor for efficiency
        
        return green_tokens, red_tokens


class LLMWatermarkerEncoder(LLMWatermarkerBase):
    def __init__(
        self,
        model_name: str,
        secret_key: str,
        line_fnc: callable,
        n: int,
        gf: object,
        green_list_fraction: float = 0.5,
        bias: float = 6.0,
        seed: int = 4242,
        cache_dir: str = paths.CACHE_DIR,
        device: Optional[str] = None,
        context_window: int = 1024,
        temperature: float = 0.0,
        hash_window: int = 1,
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
            device: Device to run model on ('cuda', 'cpu', or None for auto-detection)
            context_window: Maximum number of tokens to use as context for generation (default: 1024)
            temperature: Sampling temperature (default: 0.0 = greedy sampling, higher = more random)
            hash_window: Number of previous tokens to hash together (default: 1)
        """
        # Initialize base class
        super().__init__(model_name, secret_key, n, gf, green_list_fraction, seed, cache_dir)
        
        # Encoder-specific parameters
        self.line_fnc = line_fnc
        self.bias = bias
        self.context_window = context_window
        self.temperature = temperature
        self.hash_window = hash_window
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Load model (tokenizer already loaded by base class)
        self._load_model()
        
    def _load_model(self):
        """Load the model."""
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
        
    def _modify_logits(self, logits: torch.Tensor, token_id: int, secret_key: str, bias_type: str) -> torch.Tensor:
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
        green_tokens, red_tokens = self._get_red_green_tokens(token_id, secret_key)
        
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
    ) -> Tuple[str, Dict[str, int], List[int]]:
        """
        Generate text with Lagrange interpolation watermarking.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            verbose: Whether to show progress bar
            
        Returns:
            Tuple of (generated_text, statistics, watermark_blocks_info)
        """

        # Reset counters
        self.green_tokens_selected = 0
        self.red_tokens_selected = 0
        self.blocks_encoded = 0

        # Initialize tracking for watermark blocks
        watermark_blocks_info = []
        
        # Format the prompt for the specific model
        formatted_prompt = format_prompt_for_model(prompt, self.model_name, self.tokenizer)
        
        # Tokenize the formatted prompt
        input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)

        # Store generated ids
        generated_ids = input_ids.clone()[0].tolist()
        
        # Initialize K/V cache
        past_key_values = None
        cache_position = 0  # Track position in the cache
        
        # Calculate number of complete blocks we can generate
        if max_new_tokens % self.n != 0:
            raise ValueError(f"max_new_tokens ({max_new_tokens}) must be divisible by n ({self.n}) to ensure complete blocks.")
        num_blocks = max_new_tokens // self.n
        if num_blocks == 0:
            raise ValueError(f"max_new_tokens ({max_new_tokens}) must be at least n ({self.n}) to generate at least one block.")
        
        # Setup progress tracking
        total_tokens_to_generate = num_blocks * self.n
        progress_bar = tqdm(range(total_tokens_to_generate), disable=not verbose)
        tokens_generated = 0
        
        # Generate tokens block by block
        for block_idx in range(num_blocks):
            # At the start of each block, determine the token to use for x-coordinate computation
            if len(generated_ids) > len(input_ids[0]):
                # Use the last generated token for subsequent blocks
                previous_token_id = generated_ids[-1]
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
            
            # Track this block
            block_info = {
                "block_idx": block_idx,
                "x": int(x),
                "y": int(y),
                "y_bits": y_bits.copy(),
                "encoded_bits": [],
                "tokens": []
            }
            
            # Generate n tokens for this block (one for each bit of y)
            for bit_idx in range(self.n):
                # Prepare input for the model
                if past_key_values is None:
                    # First generation - use the full prompt
                    # Only use the last context_window tokens if needed
                    if len(generated_ids) > self.context_window:
                        model_input_ids = torch.tensor([generated_ids[-self.context_window:]], device=self.device)
                        cache_position = 0  # Reset cache position if we truncate
                    else:
                        model_input_ids = torch.tensor([generated_ids], device=self.device)
                else:
                    # Subsequent generations - only use the last generated token
                    model_input_ids = torch.tensor([[generated_ids[-1]]], device=self.device)
                    
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
                
                # Determine bias type based on current bit
                current_bit = y_bits[bit_idx]
                bias_type = "green" if current_bit == 1 else "red"
                
                # Use the previous token for vocabulary splitting
                current_previous_token = generated_ids[-1]
                
                # Modify logits with watermark using the current previous token and secret key
                modified_logits = self._modify_logits(logits, current_previous_token, self.secret_key, bias_type)
                
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
                green_tokens, red_tokens = self._get_red_green_tokens(current_previous_token, self.secret_key)
                
                # Check vocabulary bounds for safety
                vocab_size = len(self.tokenizer)
                if next_token_id < vocab_size:  # Make sure token is in vocabulary range
                    # Create a tensor from the next token ID to enable vectorized comparison
                    # Make sure it's on the same device as green_tokens
                    next_token_tensor = torch.tensor(next_token_id, device=green_tokens.device)
                    
                    # Use tensor operations to efficiently check if the token is in the green list
                    is_green = (green_tokens == next_token_tensor).any().item()
                    if is_green:
                        block_info["encoded_bits"].append(1)
                        self.green_tokens_selected += 1
                    else:
                        block_info["encoded_bits"].append(0)
                        self.red_tokens_selected += 1

                # Add the new token to generated ids
                generated_ids.append(next_token_id)
                block_info['tokens'].append(next_token_id)
                tokens_generated += 1
                
                # Update progress bar with stats
                progress_bar.set_description(f"Block {block_idx+1}/{num_blocks}, Bit {bit_idx+1}/{self.n}, Green: {self.green_tokens_selected}, Red: {self.red_tokens_selected}")
                progress_bar.update(1)
                
                # Check if we've reached an EOS token
                if next_token_id == self.tokenizer.eos_token_id:
                    break
            
            # Add completed block info
            watermark_blocks_info.append(block_info)
            self.blocks_encoded += 1
            
            # Check if we hit EOS in the middle of a block
            if generated_ids[-1] == self.tokenizer.eos_token_id:
                break
        
        progress_bar.close()

        # Decode and return the generated text
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Create statistics dictionary
        statistics = {
            'green_tokens': self.green_tokens_selected,
            'red_tokens': self.red_tokens_selected,
            'blocks_encoded': self.blocks_encoded,
            'total_tokens_generated': tokens_generated,
            'green_ratio': self.green_tokens_selected / (self.green_tokens_selected + self.red_tokens_selected + 1e-10)
        }

        return generated_text, statistics, watermark_blocks_info


class LLMWatermarkerDecoder(LLMWatermarkerBase):
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
        green_list_fraction: float = 0.5,
        seed: int = 4242,
        cache_dir: str = paths.CACHE_DIR,
    ):
        """
        Initialize the decoder with the same parameters used for encoding.
        
        Args:
            model_name: HuggingFace model identifier (same as used for encoding)
            secret_key: Secret key for watermarking (same as used for encoding)
            n: Field size parameter (GF(2^n)) (same as used for encoding)
            gf: Galois field instance GF(2^n) (same as used for encoding)
            green_list_fraction: Fraction of tokens in green list (same as used for encoding)
            seed: Random seed for reproducibility (same as used for encoding)
            cache_dir: Directory to cache models
        """
        # Initialize base class (loads tokenizer)
        super().__init__(model_name, secret_key, n, gf, green_list_fraction, seed, cache_dir)
        
    def decode_text(self, text: str, prompt: str = None) -> List[Dict[str, Union[int, List[int]]]]:
        """
        Decode watermark information from generated text.
        
        Args:
            text: The generated text to decode
            prompt: Optional original prompt to exclude from decoding
            
        Returns:
            List of blocks, each containing:
            - 'x': x-coordinate (GF element as integer)
            - 'y_bits': Binary sequence representing y-value [0,1,0,1,...]
        """
        # Tokenize the input text
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        
        # If prompt is provided, extract only the generated portion
        if prompt is not None:
            from src.model_formatters import format_prompt_for_model
            formatted_prompt = format_prompt_for_model(prompt, self.model_name, self.tokenizer)
            prompt_token_ids = self.tokenizer.encode(formatted_prompt, add_special_tokens=False)
            
            # Extract only the generated tokens (after the prompt)
            if len(token_ids) > len(prompt_token_ids):
                token_ids = token_ids[len(prompt_token_ids):]
            else:
                return []  # No generated tokens to decode
        
        # Calculate number of complete blocks
        num_complete_blocks = len(token_ids) // self.n
        
        if num_complete_blocks == 0:
            return []  # No complete blocks to decode
        
        blocks = []
        
        # Process each complete block
        for block_idx in range(num_complete_blocks):
            start_idx = block_idx * self.n
            end_idx = start_idx + self.n
            block_tokens = token_ids[start_idx:end_idx]
            
            # Determine previous token for x-coordinate calculation
            if block_idx == 0:
                # First block uses token ID 0 (same as encoder)
                previous_token = 0
            else:
                # Use last token of previous block
                previous_token = token_ids[start_idx - 1]
            
            # Calculate x-coordinate using the same method as encoder
            x = self._hash_to_gf_element(previous_token, self.secret_key)
            
            # Extract y_bits by classifying each token as green (1) or red (0)
            y_bits = []
            for i, token_id in enumerate(block_tokens):
                # Determine the previous token for vocabulary splitting
                if start_idx + i == 0:
                    # Very first token in the sequence
                    vocab_split_token = 0
                else:
                    # Use the previous token in the sequence
                    vocab_split_token = token_ids[start_idx + i - 1]
                
                # Get green and red token lists for this position
                green_tokens, red_tokens = self._get_red_green_tokens(vocab_split_token, self.secret_key)
                
                # Check if current token is in green list (1) or red list (0)
                is_green = (green_tokens == token_id).any().item()
                y_bits.append(1 if is_green else 0)
            
            # Store the extracted block information
            blocks.append({
                'x': int(x),
                'y_bits': y_bits
            })
        
        return blocks
