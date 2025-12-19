"""
Hamming Code Implementation for Watermark Error Detection/Correction

Supports:
- Standard Hamming (d_min=3): Correct 1-bit errors
- SECDED (d_min=4): Correct 1-bit errors, detect 2-bit errors

Uses SYSTEMATIC format: codeword = [data_bits | parity_bits]

Usage:
    # Initialize once for a given block size
    hamming = HammingCode(n=8, secded=False)

    # Reuse for multiple blocks
    for block in blocks:
        codeword, p_bits = hamming.encode(block)
        ...
        decoded, syndrome, valid = hamming.decode(received)
"""

from typing import List, Tuple


class HammingCode:
    """
    Hamming code encoder/decoder - initialize once, reuse for all blocks.

    Systematic codeword structure:
        [D0, D1, ..., D_{n-1}, P0, P1, ..., P_{r-1}, (P_overall for SECDED)]
        |______ data ________|  |_______ parity _______|

    Example for n=8, r=4:
        Index:   0   1   2   3   4   5   6   7   8   9  10  11
        Type:   D0  D1  D2  D3  D4  D5  D6  D7  P1  P2  P4  P8

    Parity bit coverage (based on logical positions):
        P1 covers data bits where logical_position & 1: D0, D1, D3, D4, D6
        P2 covers data bits where logical_position & 2: D0, D2, D3, D5, D6
        P4 covers data bits where logical_position & 4: D1, D2, D3, D7
        P8 covers data bits where logical_position & 8: D4, D5, D6, D7
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

        # Logical positions (1-indexed, used for parity coverage calculation)
        # Parity positions: 1, 2, 4, 8, ...
        self.parity_positions = tuple(2**i for i in range(self.r))

        # Data positions: all non-power-of-2 up to n+r
        parity_set = set(self.parity_positions)
        self.data_positions = tuple(
            i for i in range(1, n + self.r + 1)
            if i not in parity_set
        )

        # Pre-compute parity coverage: which data bits each parity bit covers
        # coverage[j] = list of data indices covered by parity bit j
        self._parity_coverage = []
        for p_pos in self.parity_positions:
            coverage = [i for i, d_pos in enumerate(self.data_positions) if d_pos & p_pos]
            self._parity_coverage.append(coverage)

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

    def encode(self, data: List[int]) -> Tuple[List[int], List[int]]:
        """
        Encode n data bits into systematic codeword.

        Args:
            data: List of n bits [d0, d1, ..., d_{n-1}]

        Returns:
            Tuple of (codeword, p_bits):
            - codeword: [data | parity] as list of bits
            - p_bits: parity bits only (for easy access)
        """
        if len(data) != self.n:
            raise ValueError(f"Expected {self.n} data bits, got {len(data)}")

        # Compute parity bits using pre-computed coverage
        p_bits = []
        for coverage in self._parity_coverage:
            parity = 0
            for i in coverage:
                parity ^= data[i]
            p_bits.append(parity)

        # SECDED: compute overall parity
        if self.secded:
            overall = 0
            for bit in data:
                overall ^= bit
            for bit in p_bits:
                overall ^= bit
            p_bits.append(overall)

        # Systematic codeword: data followed by parity
        codeword = list(data) + p_bits

        return codeword, p_bits

    def decode(self, codeword: List[int], correct: bool = False) -> Tuple[List[int], int, bool]:
        """
        Decode systematic codeword, detect/correct errors.

        Args:
            codeword: Systematic codeword [data | parity]
            correct: If True, attempt error correction. If False, detection-only mode
                     (better filtering with lower false positive rate).

        Returns:
            Tuple of (data_bits, syndrome, is_valid):
            - data_bits: n data bits (potentially corrected if correct=True)
            - syndrome: error location in logical position (0 = no error)
            - is_valid: True if no error detected (or corrected if correct=True)
        """
        expected_len = self.codeword_length
        if len(codeword) != expected_len:
            raise ValueError(f"Expected {expected_len} bits, got {len(codeword)}")

        # Split systematic codeword
        data_bits = list(codeword[:self.n])
        parity_bits = codeword[self.n:]

        # Handle SECDED overall parity
        overall_parity = 0
        if self.secded:
            for bit in codeword:
                overall_parity ^= bit

        # Calculate syndrome using pre-computed coverage
        syndrome = 0
        for j, (p_pos, coverage) in enumerate(zip(self.parity_positions, self._parity_coverage)):
            expected_parity = 0
            for i in coverage:
                expected_parity ^= data_bits[i]
            if expected_parity != parity_bits[j]:
                syndrome |= p_pos

        # Determine validity and correct if possible
        is_valid = True

        if self.secded:
            if syndrome == 0 and overall_parity == 0:
                pass  # No error
            elif syndrome == 0 and overall_parity == 1:
                # Error in overall parity bit only - data is fine
                if not correct:
                    is_valid = False  # Strict detection: any parity error = invalid
            elif syndrome > 0 and overall_parity == 1:
                # Single-bit error (correctable)
                if correct:
                    data_bits = self._correct_error(data_bits, syndrome)
                else:
                    is_valid = False
            else:  # syndrome > 0 and overall_parity == 0
                # Double-bit error detected (not correctable)
                is_valid = False
        else:
            # Standard Hamming
            if syndrome > 0:
                if correct:
                    data_bits = self._correct_error(data_bits, syndrome)
                else:
                    is_valid = False

        return data_bits, syndrome, is_valid

    def _correct_error(self, data_bits: List[int], syndrome: int) -> List[int]:
        """
        Correct single-bit error based on syndrome.

        Args:
            data_bits: Data bits (will be modified if error is in data)
            syndrome: Logical position of error

        Returns:
            Corrected data bits
        """
        # Check if error is in a data position
        for i, d_pos in enumerate(self.data_positions):
            if d_pos == syndrome:
                data_bits[i] ^= 1
                return data_bits

        # Error is in a parity position - data is unaffected
        return data_bits

    def __repr__(self) -> str:
        mode = "SECDED" if self.secded else "Standard"
        return f"HammingCode(n={self.n}, r={self.r}, mode={mode}, codeword={self.codeword_length}, systematic=True)"
