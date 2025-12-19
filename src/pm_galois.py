"""
Flexible Galois Field GF(2^n) implementation for watermark verification.
Supports any field size with intelligent optimization strategies.

Based on: https://github.com/pawelmorawiecki/Maximum_Collinear_Points/blob/main/maximum_collinear.py
"""

import random
from typing import Dict, Optional, Tuple, List

class GaloisField:
    """
    Flexible GF(2^n) implementation that adapts optimization strategies based on field size.
    Works directly with integers for maximum compatibility and performance.
    """
    
    # Irreducible polynomials matching the galois package (https://mhostetter.github.io/galois/)
    # These define the field arithmetic - must be consistent across encoder/decoder
    # Format: n -> polynomial (as integer with bit representation)
    IRREDUCIBLE_POLYNOMIALS = {
        1: 0x3,       # x + 1
        2: 0x7,       # x^2 + x + 1
        3: 0xB,       # x^3 + x + 1
        4: 0x13,      # x^4 + x + 1
        5: 0x25,      # x^5 + x^2 + 1
        6: 0x5B,      # x^6 + x^4 + x^3 + x + 1
        7: 0x83,      # x^7 + x + 1
        8: 0x11D,     # x^8 + x^4 + x^3 + x^2 + 1
        9: 0x211,     # x^9 + x^4 + 1
        10: 0x46F,    # x^10 + x^6 + x^5 + x^3 + x^2 + x + 1
        11: 0x805,    # x^11 + x^2 + 1
        12: 0x10EB,   # x^12 + x^7 + x^6 + x^5 + x^3 + x + 1
        13: 0x201B,   # x^13 + x^4 + x^3 + x + 1
        14: 0x40A9,   # x^14 + x^7 + x^5 + x^3 + 1
        15: 0x8035,   # x^15 + x^5 + x^4 + x^2 + 1
        16: 0x1002D,   # x^16 + x^5 + x^3 + x^2 + 1
        17: 0x20009,   # x^17 + x^3 + 1
        18: 0x41403,   # x^18 + x^12 + x^10 + x + 1
        19: 0x80027,   # x^19 + x^5 + x^2 + x + 1
        20: 0x1006F3,  # x^20 + x^10 + x^9 + x^7 + x^6 + x^5 + x^4 + x + 1
        21: 0x200065,  # x^21 + x^6 + x^5 + x^2 + 1
        22: 0x401F61,  # x^22 + x^12 + x^11 + x^10 + x^9 + x^8 + x^6 + x^5 + 1
        23: 0x800021,  # x^23 + x^5 + 1
        24: 0x101E6A9, # x^24 + x^16 + x^15 + x^14 + x^13 + x^9 + x^7 + x^5 + x^3 + 1
        25: 0x2000145, # x^25 + x^8 + x^6 + x^2 + 1
        26: 0x40045D3, # x^26 + x^14 + x^10 + x^8 + x^7 + x^6 + x^4 + x + 1
    }
    
    def __init__(self, n: int):
        """
        Initialize GF(2^n) field.
        
        Args:
            n: Field size parameter (creates GF(2^n))
        """
        if n < 1:
            raise ValueError("Field size parameter n must be at least 1")
        
        self.n = n
        self.field_size = 2 ** n
        self.field_mask = self.field_size - 1  # Mask for n bits
        
        # Get irreducible polynomial
        if n in self.IRREDUCIBLE_POLYNOMIALS:
            self.mod_poly = self.IRREDUCIBLE_POLYNOMIALS[n]
        else:
            # For unsupported sizes, use a simple polynomial (may not be irreducible)
            # In practice, you'd want to find proper irreducible polynomials
            self.mod_poly = (1 << n) | 1  # x^n + 1 (not always irreducible)
        
        # Decide on optimization strategy based on field size
        self.use_precomputed_inverses = n <= 20  # Don't precompute above 2^20
        
        if self.use_precomputed_inverses:
            self.inv_table = self._precompute_inverses()
        else:
            self.inv_table = None
    
    def _precompute_inverses(self) -> List[int]:
        """Pre-compute inverse table for all non-zero elements."""
        inv_table = [0] * self.field_size
        for v in range(1, self.field_size):
            inv_table[v] = self._compute_inverse(v)
        return inv_table
    
    def _compute_inverse(self, a: int) -> int:
        """
        Compute inverse of element a using Extended Euclidean algorithm.
        
        Args:
            a: Element to find inverse of
            
        Returns:
            Inverse of a in GF(2^n)
        """
        if a == 0:
            raise ZeroDivisionError("Inverse of zero does not exist")
        
        lm, hm = 1, 0
        low, high = a, self.mod_poly
        
        while low > 1:
            shift = high.bit_length() - low.bit_length()
            if shift < 0:
                # Swap low<->high, lm<->hm
                low, high = high, low
                lm, hm = hm, lm
                shift = -shift
            
            # high = high + (low << shift)  [+ = xor in GF(2)]
            high ^= low << shift
            hm ^= lm << shift
        
        return lm & self.field_mask
    
    def add(self, a: int, b: int) -> int:
        """
        Addition in GF(2^n): simple XOR.
        
        Args:
            a, b: Elements to add
            
        Returns:
            Sum in GF(2^n)
        """
        return (a ^ b) & self.field_mask
    
    def multiply(self, a: int, b: int) -> int:
        """
        Multiplication in GF(2^n) modulo the irreducible polynomial.
        
        Args:
            a, b: Elements to multiply
            
        Returns:
            Product in GF(2^n)
        """
        a &= self.field_mask
        b &= self.field_mask
        
        result = 0
        while b:
            if b & 1:
                result ^= a
            b >>= 1
            a <<= 1
            # If degree >= n, reduce
            if a & (1 << self.n):
                a ^= self.mod_poly
        
        return result & self.field_mask
    
    def divide(self, a: int, b: int) -> int:
        """
        Division in GF(2^n).
        
        Args:
            a: Dividend
            b: Divisor
            
        Returns:
            Quotient a/b in GF(2^n)
        """
        if b == 0:
            raise ZeroDivisionError("Division by zero in GF(2^n)")
        
        return self.multiply(a, self.inverse(b))
    
    def inverse(self, a: int) -> int:
        """
        Get multiplicative inverse of element a.
        
        Args:
            a: Element to find inverse of
            
        Returns:
            Inverse of a in GF(2^n)
        """
        if a == 0:
            raise ZeroDivisionError("Inverse of zero does not exist")
        
        a &= self.field_mask
        
        if self.use_precomputed_inverses:
            return self.inv_table[a]
        else:
            return self._compute_inverse(a)
    
    def subtract(self, a: int, b: int) -> int:
        """
        Subtraction in GF(2^n): same as addition (XOR).
        
        Args:
            a, b: Elements to subtract
            
        Returns:
            Difference a-b in GF(2^n)
        """
        return self.add(a, b)  # In GF(2), subtraction = addition
    
    def power(self, a: int, exp: int) -> int:
        """
        Exponentiation in GF(2^n).
        
        Args:
            a: Base
            exp: Exponent
            
        Returns:
            a^exp in GF(2^n)
        """
        if exp == 0:
            return 1
        if exp == 1:
            return a & self.field_mask
        
        result = 1
        base = a & self.field_mask
        
        while exp > 0:
            if exp & 1:
                result = self.multiply(result, base)
            base = self.multiply(base, base)
            exp >>= 1
        
        return result


def max_collinear_points(points: List[Tuple[int, int]], gf: GaloisField) -> Tuple[int, Optional[int], List[Tuple[int, int]]]:
    """
    Find maximum number of collinear points using the provided Galois field.
    
    Args:
        points: List of (x, y) tuples as integers
        gf: GaloisField instance to use for calculations
        
    Returns:
        Tuple of (max_count, best_slope, collinear_points)
    """
    n = len(points)
    if n < 2:
        return 0 if n == 0 else 1, None, points.copy()
    
    best_count = 1
    best_slope = None
    best_collinear_points = []
    
    # For each point i, count slopes to points j > i
    for i in range(n):
        xi, yi = points[i]
        slope_counts = {}
        slope_points = {}  # Track which points belong to each slope
        
        for j in range(i + 1, n):
            xj, yj = points[j]
            
            dx = gf.subtract(xi, xj)  # xi - xj in GF(2^n)
            dy = gf.subtract(yi, yj)  # yi - yj in GF(2^n)
            
            if dx == 0:
                # Vertical line - skip as mentioned in the paper
                continue
            else:
                # slope = dy / dx
                slope = gf.divide(dy, dx)
            
            # Count points for this slope
            if slope not in slope_counts:
                slope_counts[slope] = 0
                slope_points[slope] = []
            
            slope_counts[slope] += 1
            slope_points[slope].append((xj, yj))
        
        # Check if any slope from point i gives a better result
        for slope, count in slope_counts.items():
            total_count = count + 1  # +1 for point i itself
            if total_count > best_count:
                best_count = total_count
                best_slope = slope
                # Include point i and all points with this slope
                best_collinear_points = [(xi, yi)] + slope_points[slope]
    
    return best_count, best_slope, best_collinear_points


def recover_line_equation(collinear_points: List[Tuple[int, int]], gf: GaloisField) -> Tuple[int, int]:
    """
    Recover line equation f(x) = a₀ + a₁x from collinear points.
    
    Args:
        collinear_points: List of collinear points as integer tuples
        gf: GaloisField instance to use for calculations
        
    Returns:
        Tuple of (a₀, a₁) as integers
    """
    if len(collinear_points) < 2:
        raise ValueError("Need at least 2 points to determine a line")
    
    # Take the first two points to determine the line
    (x1, y1) = collinear_points[0]
    (x2, y2) = collinear_points[1]
    
    dx = gf.subtract(x2, x1)  # x2 - x1
    if dx == 0:
        raise ValueError("Cannot determine line from vertical points")
    
    # Calculate slope: a₁ = (y₂ - y₁) / (x₂ - x₁)
    dy = gf.subtract(y2, y1)  # y2 - y1
    a1 = gf.divide(dy, dx)
    
    # Calculate intercept: a₀ = y₁ - a₁ * x₁
    a1_x1 = gf.multiply(a1, x1)
    a0 = gf.subtract(y1, a1_x1)  # y1 - (a1 * x1)
    
    return a0, a1


# Test function for verification
def test_galois_field():
    """Test the GaloisField implementation with different field sizes."""
    print("Testing GaloisField implementation...")
    
    # Test different field sizes
    for n in [4, 8, 16]:
        print(f"\nTesting GF(2^{n})...")
        gf = GaloisField(n)
        
        # Test basic operations
        a, b = 5, 3
        if a >= gf.field_size:
            a = a % gf.field_size
        if b >= gf.field_size:
            b = b % gf.field_size
        
        print(f"  Field size: {gf.field_size}")
        print(f"  a = {a}, b = {b}")
        print(f"  a + b = {gf.add(a, b)}")
        print(f"  a * b = {gf.multiply(a, b)}")
        
        if b != 0:
            inv_b = gf.inverse(b)
            print(f"  b^(-1) = {inv_b}")
            print(f"  a / b = {gf.divide(a, b)}")
            # Verify: b * b^(-1) = 1
            product = gf.multiply(b, inv_b)
            print(f"  Verification: b * b^(-1) = {product} (should be 1)")
        
        # Test with random points
        points = [(random.randint(0, gf.field_size-1), random.randint(0, gf.field_size-1)) for _ in range(10)]
        max_count, best_slope, collinear_points = max_collinear_points(points, gf)
        print(f"  Max collinear points: {max_count}")
        
        if max_count >= 2:
            try:
                a0, a1 = recover_line_equation(collinear_points, gf)
                print(f"  Recovered line: f(x) = {a0} + {a1}*x")
            except ValueError as e:
                print(f"  Could not recover line: {e}")
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_galois_field()
