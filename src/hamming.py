"""
Hamming Code Implementation for Watermark Error Detection/Correction

Supports:
- Standard Hamming (d_min=3): Correct 1-bit errors
- SECDED (d_min=4): Correct 1-bit errors, detect 2-bit errors

Usage:
    # Initialize once for a given block size
    hamming = HammingCode(n=8, secded=False)

    # Reuse for multiple blocks
    for block in blocks:
        encoded = hamming.encode(block)
        ...
        decoded, syndrome, valid = hamming.decode(received)
"""

from typing import List, Tuple


class HammingCode:
    """
    Hamming code encoder/decoder - initialize once, reuse for all blocks.

    Codeword structure (1-indexed positions):
    - Parity bits at positions 1, 2, 4, 8, 16, ... (powers of 2)
    - Data bits fill remaining positions: 3, 5, 6, 7, 9, 10, 11, ...

    Example for n=8, r=4 (positions 1-12):
        Position:  1   2   3   4   5   6   7   8   9  10  11  12
        Type:     P1  P2  D0  P4  D1  D2  D3  P8  D4  D5  D6  D7
    """

    def __init__(self, n: int, secded: bool = False):
        """
        Initialize Hamming code for given data size.

        Args:
            n: Number of data bits per block (same as GF(2^n) field size parameter)
            secded: If True, add overall parity for double-error detection
        """
        self.n = n
        self.secded = secded

        # Compute r: smallest r where 2^r >= n + r + 1
        self.r = self._compute_parity_bits(n)

        # Pre-compute positions (reused for every encode/decode)
        # Parity positions (1-indexed): 1, 2, 4, 8, ...
        self.parity_positions = tuple(2**i for i in range(self.r))

        # Data positions (1-indexed): all non-power-of-2 up to n+r
        parity_set = set(self.parity_positions)
        self.data_positions = tuple(
            i for i in range(1, n + self.r + 1)
            if i not in parity_set
        )

    @property
    def parity_bit_count(self) -> int:
        """Total parity bits: r for standard, r+1 for SECDED."""
        return self.r + (1 if self.secded else 0)

    @property
    def codeword_length(self) -> int:
        """Total codeword length: n + r (+ 1 for SECDED)."""
        return self.n + self.parity_bit_count

    def _compute_parity_bits(self, n: int) -> int:
        """Find smallest r where 2^r >= n + r + 1."""
        r = 1
        while (2 ** r) < (n + r + 1):
            r += 1
        return r

    def encode(self, data: List[int]) -> List[int]:
        """
        Encode n data bits into (n+r) or (n+r+1) codeword.

        Args:
            data: List of n bits [d0, d1, ..., d_{n-1}]

        Returns:
            Codeword as list of bits
        """
        if len(data) != self.n:
            raise ValueError(f"Expected {self.n} data bits, got {len(data)}")

        # Create codeword array (use 0-indexed, but positions are 1-indexed)
        codeword = [0] * (self.n + self.r)

        # Place data bits in data positions
        for i, pos in enumerate(self.data_positions):
            codeword[pos - 1] = data[i]

        # Calculate each parity bit
        for p_pos in self.parity_positions:
            parity = 0
            for pos in range(1, len(codeword) + 1):
                if pos & p_pos:
                    parity ^= codeword[pos - 1]
            codeword[p_pos - 1] = parity

        # SECDED: append overall parity
        if self.secded:
            overall = 0
            for bit in codeword:
                overall ^= bit
            codeword.append(overall)

        return codeword

    def decode(self, codeword: List[int]) -> Tuple[List[int], int, bool]:
        """
        Decode codeword, detect/correct errors.

        Args:
            codeword: List of (n+r) or (n+r+1) bits

        Returns:
            Tuple of (corrected_data, syndrome, is_valid):
            - corrected_data: n data bits (potentially corrected)
            - syndrome: error position (0 = no error detected by parity checks)
            - is_valid: True if no error or correctable single-bit error
        """
        expected_len = self.codeword_length
        if len(codeword) != expected_len:
            raise ValueError(f"Expected {expected_len} bits, got {len(codeword)}")

        # Work with copy
        bits = list(codeword)

        # Handle SECDED overall parity
        overall_parity = 0
        if self.secded:
            for bit in bits:
                overall_parity ^= bit
            bits = bits[:-1]

        # Calculate syndrome
        syndrome = 0
        for p_pos in self.parity_positions:
            parity = 0
            for pos in range(1, len(bits) + 1):
                if pos & p_pos:
                    parity ^= bits[pos - 1]
            if parity:
                syndrome |= p_pos

        # Determine validity and correct if possible
        is_valid = True

        if self.secded:
            if syndrome == 0 and overall_parity == 0:
                pass  # No error
            elif syndrome == 0 and overall_parity == 1:
                pass  # Error in overall parity bit only, data unaffected
            elif syndrome > 0 and overall_parity == 1:
                # Single-bit error, correct it
                if syndrome <= len(bits):
                    bits[syndrome - 1] ^= 1
            else:  # syndrome > 0 and overall_parity == 0
                # Double-bit error detected
                is_valid = False
        else:
            # Standard Hamming: correct single-bit errors
            if syndrome > 0 and syndrome <= len(bits):
                bits[syndrome - 1] ^= 1

        # Extract data bits from corrected codeword
        data = [bits[pos - 1] for pos in self.data_positions]

        return data, syndrome, is_valid

    def __repr__(self) -> str:
        mode = "SECDED" if self.secded else "Standard"
        return f"HammingCode(n={self.n}, r={self.r}, mode={mode}, codeword={self.codeword_length})"
