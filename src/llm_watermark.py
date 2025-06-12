#!/usr/bin/env python3
"""
LLM Watermarking Implementation based on Kirchenbauer et al., 2023
https://arxiv.org/abs/2301.10226

This script implements the red-green token watermarking technique with greedy sampling.
Green tokens are used to encode
"""

import hashlib
import random
import torch
import numpy as np
import time
from typing import List, Tuple, Dict, Optional, Union
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



class LLMWatermarker:
    def __init__(
        self,
        model_name: str,
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
            green_list_fraction: Fraction of tokens to include in green list (default: 0.5)
            bias: Logit bias to apply to green tokens (default: 6.0)
            seed: Random seed for reproducibility
            cache_dir: Directory to cache models
            device: Device to run model on ('cuda', 'cpu', or None for auto-detection)
            context_window: Maximum number of tokens to use as context for generation (default: 1024)
            temperature: Sampling temperature (default: 0.0 = greedy sampling, higher = more random)
            hash_window: Number of previous tokens to hash together (default: 1)
        """
        self.model_name = model_name
        self.green_list_fraction = green_list_fraction
        self.bias = bias
        self.seed = seed
        self.cache_dir = cache_dir
        self.context_window = context_window
        self.temperature = temperature
        self.hash_window = hash_window
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Initialize random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
        # Load model and tokenizer
        self._load_model_and_tokenizer()
        
        # Stats
        self.green_tokens_selected = 0
        self.red_tokens_selected = 0
        
    def _load_model_and_tokenizer(self):
        """Load the model and tokenizer."""
        print(f"Loading model: {self.model_name}")
        
        # Get HuggingFace token if available
        token = load_hf_token()
        
        # Configure tokenizer options
        tokenizer_kwargs = {
            "cache_dir": paths.MODELS_CACHE_DIR,  # Use models subdirectory
        }
        if token:
            # Use token instead of deprecated use_auth_token
            tokenizer_kwargs["token"] = token
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            **tokenizer_kwargs
        )
        
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
            
        # Ensure the tokenizer has a padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
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
        
    def _get_red_green_tokens(self, token_ids: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate red and green token lists based on the hash of previous tokens.
        This creates a deterministic division of the vocabulary into "green" tokens
        (which receive a bias during generation) and "red" tokens (which don't).
        
        Args:
            token_ids: List of token IDs to hash (e.g., [101, 256, 512])
            
        Returns:
            Tuple of (green_tokens, red_tokens) as tensors on the device
        """
        # Convert token_ids list to a PyTorch tensor for efficient processing
        # Example: if token_ids is [101, 256, 512], token_ids_tensor becomes tensor([101, 256, 512])
        token_ids_tensor = torch.tensor(token_ids, dtype=torch.int64)
        
        # Create a hash from the specified window of tokens by joining them with hyphens
        # Example: if token_ids_tensor is tensor([101, 256, 512]), hash_input becomes "101-256-512"
        hash_input = "".join([str(tid.item()) + "-" for tid in token_ids_tensor]).rstrip("-")
        hash_object = hashlib.sha256(hash_input.encode())
        hash_hex = hash_object.hexdigest()
        
        # Convert the hexadecimal hash to a 32-bit integer for use as a random seed
        # This ensures deterministic outcomes for the same input token sequence
        hash_seed = int(hash_hex, 16) % (2**32)
        
        # Get the total number of tokens in the model's vocabulary
        vocab_size = len(self.tokenizer)
        
        # Create a tensor containing all possible token indices (0 to vocab_size-1)
        # Example: if vocab_size is 3000, all_tokens becomes tensor([0, 1, 2, ..., 2999])
        all_tokens = torch.arange(vocab_size, device=self.device)
        
        # PyTorch doesn't have a direct seeded shuffle operation, so we implement
        # a deterministic shuffle using random values and sorting
        
        # Set PyTorch's random generator to use our hash-derived seed
        # This makes the shuffling deterministic based on the input tokens
        torch.manual_seed(hash_seed)
        
        # Create random values and sort them to generate a permutation of indices
        # Example: random_values might be tensor([0.12, 0.95, 0.33, ...])
        random_values = torch.rand(vocab_size, device=self.device)
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
        
    def _modify_logits(self, logits: torch.Tensor, token_window: List[int]) -> torch.Tensor:
        """
        Modify logits by adding bias to green tokens.
        This is the core watermarking function that biases the model's predictions
        toward selecting tokens from the "green list" determined by previous tokens.
        
        Args:
            logits: Original logits from the model (prediction scores for each token)
            token_window: List of previous token IDs to use for hashing (e.g., [101, 256, 512])
            
        Returns:
            Modified logits tensor with bias added to green tokens
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
        
        # Get green and red tokens as tensors using the deterministic hash of previous tokens
        # Example: based on token_window [101, 256, 512], this might return
        # green_tokens as tensor([42, 900, 5, ...]) and red_tokens (which we don't use here)
        green_tokens, _ = self._get_red_green_tokens(token_window)
        
        # Clone logits for modification to avoid affecting the original tensor
        # This creates a new tensor with the same values that we can safely modify
        modified_logits = logits.clone()
        
        # Filter green tokens to ensure they're within vocabulary bounds using tensor operations
        # Example: if vocab_size is 50,000 and green_tokens contains a value 60,000,
        # we create a boolean mask tensor([True, True, True, False, ...]) for values < 50,000
        mask = green_tokens < vocab_size
        # Apply the mask to get only valid green tokens
        # Example: valid_green_tokens becomes tensor([42, 900, 5, ...]) without any out-of-bounds tokens
        valid_green_tokens = green_tokens[mask]
        
        # Vectorized bias application - apply bias to all valid green tokens at once
        # This means tokens in the "green list" will have higher probability of being selected
        if valid_green_tokens.numel() > 0:  # Only proceed if we have valid tokens
            # Example: if self.bias is 6.0, this adds 6.0 to the logits for all green tokens
            # For the token indices in valid_green_tokens
            modified_logits[valid_green_tokens] += self.bias
        
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
        Generate text with watermarking.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            verbose: Whether to show progress bar
            
        Returns:
            Tuple of (generated_text, statistics, green_red_mask)
        """
        # Record start time for total duration
        total_start_time = time.time()

        # Reset counters
        self.green_tokens_selected = 0
        self.red_tokens_selected = 0

        # Initialize mask for green/red tokens
        green_red_mask = []

        # Initialize timing variables
        prompt_formatting_duration = 0.0
        tokenization_duration = 0.0
        logits_generation_time = 0.0
        modify_logits_time = 0.0
        sampling_time = 0.0
        
        # Format the prompt for the specific model
        start_time = time.time()
        formatted_prompt = format_prompt_for_model(prompt, self.model_name, self.tokenizer)
        end_time = time.time()
        prompt_formatting_duration = end_time - start_time
        
        # Tokenize the formatted prompt
        start_time = time.time()
        input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)
        end_time = time.time()
        tokenization_duration = end_time - start_time
        
        # Store generated ids
        generated_ids = input_ids.clone()[0].tolist()
        
        # Initialize K/V cache
        past_key_values = None
        cache_position = 0  # Track position in the cache
        
        # Setup progress tracking
        progress_bar = tqdm(range(max_new_tokens), disable=not verbose)
        
        # Generate tokens one by one
        for i in progress_bar:
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
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model(
                    model_input_ids, 
                    past_key_values=past_key_values,
                    use_cache=True  # Enable K/V caching
                )
                logits = outputs.logits[:, -1:, :]  # Get logits of last token
                past_key_values = outputs.past_key_values  # Update cache
                cache_position += model_input_ids.shape[1]  # Update position
            end_time = time.time()
            logits_generation_time += end_time - start_time
            
            # Create a window of previous tokens for hashing
            if len(generated_ids) >= self.hash_window:
                token_window = generated_ids[-self.hash_window:]
            else:
                token_window = generated_ids
            
            # Modify logits with watermark using the token window
            start_time = time.time()
            modified_logits = self._modify_logits(logits, token_window)
            end_time = time.time()
            modify_logits_time += end_time - start_time
            
            # Apply temperature if set
            if self.temperature > 0:
                # Scale logits by temperature
                modified_logits = modified_logits / self.temperature
                
            # Get probabilities through softmax
            probs = torch.nn.functional.softmax(modified_logits, dim=-1)
            
            # Token sampling based on temperature
            start_time = time.time()
            if self.temperature == 0 or self.temperature < 1e-6:
                # Greedy sampling (select token with highest probability)
                next_token_id = torch.argmax(probs, dim=-1).item()
            else:
                # Sample from the distribution
                next_token_id = torch.multinomial(probs.squeeze(), 1).item()
            end_time = time.time()
            sampling_time += end_time - start_time
            
            # Track whether the selected token was from the green or red list for watermark statistics
            # Get the current division of tokens into green and red based on previous tokens
            green_tokens, red_tokens = self._get_red_green_tokens(token_window)
            
            # Check vocabulary bounds for safety
            vocab_size = len(self.tokenizer)
            if next_token_id < vocab_size:  # Make sure token is in vocabulary range
                # Create a tensor from the next token ID to enable vectorized comparison
                # Example: if next_token_id is 42, this becomes tensor(42) on the device
                next_token_tensor = torch.tensor(next_token_id, device=self.device)
                
                # Use tensor operations to efficiently check if the token is in the green list
                # Example: if green_tokens contains 42, this returns True, otherwise False
                is_green = (green_tokens == next_token_tensor).any().item()
                
                # Update statistics counters based on whether the selected token was green or red
                if is_green:
                    self.green_tokens_selected += 1  # Token was from the green list (biased)
                    green_red_mask.append(1)
                else:
                    self.red_tokens_selected += 1    # Token was from the red list (unbiased)
                    green_red_mask.append(0)
            
            # Update progress bar with stats
            green_ratio = self.green_tokens_selected / (self.green_tokens_selected + self.red_tokens_selected + 1e-10)
            progress_bar.set_description(f"Green: {self.green_tokens_selected}, Red: {self.red_tokens_selected}, Ratio: {green_ratio:.2f}")
            
            # Add the new token to generated ids
            generated_ids.append(next_token_id)
            
            # Check if we've reached an EOS token
            if next_token_id == self.tokenizer.eos_token_id:
                break
                
        # Record end time for total duration
        total_end_time = time.time()
        total_duration = total_end_time - total_start_time

        # Decode and return the generated text
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Compile statistics
        stats = {
            "green_tokens": self.green_tokens_selected,
            "red_tokens": self.red_tokens_selected,
            "total_tokens": self.green_tokens_selected + self.red_tokens_selected,
            "green_ratio": self.green_tokens_selected / (self.green_tokens_selected + self.red_tokens_selected + 1e-10)
        }

        # Calculate time per token
        total_tokens = stats['total_tokens']
        if total_tokens > 0:
            prompt_formatting_per_token = prompt_formatting_duration / total_tokens
            tokenization_per_token = tokenization_duration / total_tokens
            logits_generation_per_token = logits_generation_time / total_tokens
            modify_logits_per_token = modify_logits_time / total_tokens
            sampling_per_token = sampling_time / total_tokens
            total_per_token = total_duration / total_tokens
        else:
            prompt_formatting_per_token = 0.0
            tokenization_per_token = 0.0
            logits_generation_per_token = 0.0
            modify_logits_per_token = 0.0
            sampling_per_token = 0.0
            total_per_token = 0.0

        # Print timing summary
        print("\n--- Timing Summary (s) ---")
        print(f"{'':<30} {'total':>10}  |  {'per token'}")
        print(f"{'prompt formatting:':<30} {prompt_formatting_duration:>10.2f}  |  {prompt_formatting_per_token:.4f}")
        print(f"{'tokenization:':<30} {tokenization_duration:>10.2f}  |  {tokenization_per_token:.4f}")
        print(f"{'logits generation:':<30} {logits_generation_time:>10.2f}  |  {logits_generation_per_token:.4f}")
        print(f"{'modify logits (watermarking):':<30} {modify_logits_time:>10.2f}  |  {modify_logits_per_token:.4f}")
        print(f"{'sampling:':<30} {sampling_time:>10.2f}  |  {sampling_per_token:.4f}")
        print(f"{'total time:':<30} {total_duration:>10.2f}  |  {total_per_token:.4f}")
        print("----------------------\n")

        return generated_text, stats, green_red_mask
