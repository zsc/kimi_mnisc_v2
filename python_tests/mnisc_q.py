"""
MNISC-Q Quantization Codec Module

MNISC-Q编码规则:
- 2-bit code -> value: 00->-3, 01->-1, 10->+1, 11->+3
- int4 (4-bit): 两个2-bit slice组合: val = decode2(slice0) + (decode2(slice1) << 2)
- int8 (8-bit): 四个2-bit slice组合: val = Σ decode2(slice_s) << (2*s)
"""

import numpy as np
from typing import List, Tuple

# MNISC-Q 2-bit decode lookup table
# code: 00->-3, 01->-1, 10->+1, 11->+3
DECODE2_TABLE = np.array([-3, -1, 1, 3], dtype=np.int32)
ENCODE2_TABLE = { -3: 0, -1: 1, 1: 2, 3: 3 }


def decode2(code: int) -> int:
    """2-bit code -> signed int: 00->-3, 01->-1, 10->+1, 11->+3"""
    return DECODE2_TABLE[code & 0x3]


def decode_n(code: int, n_bits: int) -> int:
    """
    N-bit MNISC-Q decode by 2-bit slices.
    
    val = Σ decode2(slice_s) << (2*s)
    where slice_s = bits[2*s+1 : 2*s]
    """
    n_slices = n_bits // 2
    val = 0
    for s in range(n_slices):
        slice_code = (code >> (2 * s)) & 0x3
        val += decode2(slice_code) << (2 * s)
    return val


def _build_encode_lut(n_bits: int) -> dict:
    """Build value -> code LUT for N-bit MNISC-Q encoding."""
    lut = {}
    for code in range(2 ** n_bits):
        val = decode_n(code, n_bits)
        lut[val] = code
    return lut


def encode_n(value: int, n_bits: int) -> int:
    """
    Value -> N-bit MNISC-Q code (nearest odd).
    
    The representable set consists of all odd values in [-(2^n_bits - 1), +(2^n_bits - 1)].
    - If value is odd and in range: use exact code
    - If value is even: round away from 0
    - Then clamp to range and encode
    """
    max_val = (1 << n_bits) - 1
    
    # Clamp to valid range first
    if value > max_val:
        value = max_val
    elif value < -max_val:
        value = -max_val
    
    # Round to nearest representable odd
    if value % 2 == 0:
        if value > 0:
            value += 1
        elif value < 0:
            value -= 1
        else:
            value = 1  # tie for 0 -> +1 or -1, choose +1
    
    # Clamp again after rounding
    value = max(-max_val, min(max_val, value))
    
    # Build LUT on first use
    lut = _build_encode_lut(n_bits)
    if value in lut:
        return lut[value]
    
    # Fallback: find closest representable value
    rep_vals = np.array(sorted(lut.keys()), dtype=np.int32)
    idx = np.searchsorted(rep_vals, value)
    if idx == 0:
        closest_val = rep_vals[0]
    elif idx >= len(rep_vals):
        closest_val = rep_vals[-1]
    else:
        # Choose closer one
        if abs(rep_vals[idx] - value) < abs(rep_vals[idx - 1] - value):
            closest_val = rep_vals[idx]
        else:
            closest_val = rep_vals[idx - 1]
    return lut[int(closest_val)]


def unpack_tensor(data: bytes, bits: int, shape: List[int]) -> np.ndarray:
    """
    Unpack bytes to numpy array of decoded values.
    
    Args:
        data: Packed bytes
        bits: Bit width per element (2, 4, 8, 16, 32)
        shape: Target tensor shape
    
    Returns:
        Numpy array of decoded int32 values
    """
    total_elems = np.prod(shape)
    
    if bits == 32:
        # Direct int32 values
        arr = np.frombuffer(data, dtype=np.int32).copy()
        return arr.reshape(shape)
    elif bits == 16:
        # Little-endian int16 -> int32
        arr = np.frombuffer(data, dtype=np.int16).copy().astype(np.int32)
        return arr.reshape(shape)
    elif bits == 8:
        # 1 byte per element, decode as MNISC-Q int8
        arr = np.frombuffer(data, dtype=np.uint8).copy()
        decoded = np.array([decode_n(int(x), 8) for x in arr], dtype=np.int32)
        return decoded.reshape(shape)
    elif bits == 4:
        # 1 byte = 2 elements
        arr = np.frombuffer(data, dtype=np.uint8).copy()
        decoded = np.zeros(total_elems, dtype=np.int32)
        for i in range(total_elems):
            byte_idx = i // 2
            is_high = i % 2
            if is_high:
                code = (arr[byte_idx] >> 4) & 0xF
            else:
                code = arr[byte_idx] & 0xF
            decoded[i] = decode_n(int(code), 4)
        return decoded.reshape(shape)
    elif bits == 2:
        # 1 byte = 4 elements
        arr = np.frombuffer(data, dtype=np.uint8).copy()
        decoded = np.zeros(total_elems, dtype=np.int32)
        for i in range(total_elems):
            byte_idx = i // 4
            shift = (i % 4) * 2
            code = (arr[byte_idx] >> shift) & 0x3
            decoded[i] = decode2(int(code))
        return decoded.reshape(shape)
    else:
        raise ValueError(f"Unsupported bits: {bits}")


def pack_tensor(arr: np.ndarray, bits: int) -> bytes:
    """
    Pack numpy array to MNISC-Q coded bytes.
    
    Args:
        arr: Numpy array of values to encode
        bits: Target bit width per element
    
    Returns:
        Packed bytes
    """
    flat = arr.flatten()
    total_elems = len(flat)
    
    if bits == 32:
        # Direct int32
        return flat.astype(np.int32).tobytes()
    elif bits == 16:
        # Clip and convert to int16
        clipped = np.clip(flat, -32767, 32767).astype(np.int16)
        return clipped.tobytes()
    elif bits == 8:
        # Encode each value to MNISC-Q int8
        codes = np.array([encode_n(int(x), 8) for x in flat], dtype=np.uint8)
        return codes.tobytes()
    elif bits == 4:
        # 2 elements per byte
        num_bytes = (total_elems + 1) // 2
        result = np.zeros(num_bytes, dtype=np.uint8)
        for i in range(total_elems):
            code = encode_n(int(flat[i]), 4)
            byte_idx = i // 2
            if i % 2 == 0:
                result[byte_idx] |= code & 0xF  # Low nibble
            else:
                result[byte_idx] |= (code & 0xF) << 4  # High nibble
        return result.tobytes()
    elif bits == 2:
        # 4 elements per byte
        num_bytes = (total_elems + 3) // 4
        result = np.zeros(num_bytes, dtype=np.uint8)
        for i in range(total_elems):
            code = encode_n(int(flat[i]), 2)
            byte_idx = i // 4
            shift = (i % 4) * 2
            result[byte_idx] |= (code & 0x3) << shift
        return result.tobytes()
    else:
        raise ValueError(f"Unsupported bits: {bits}")


def requantize(x: np.ndarray, out_bits: int, saturate: bool = True) -> np.ndarray:
    """
    Requantize int32 values to N-bit MNISC-Q code.
    
    Steps:
    1. If saturate: clamp to representable range
    2. Round to nearest representable odd (ties away from 0)
    3. Encode to MNISC-Q code
    4. Decode back to value (for reference implementation)
    
    Args:
        x: Input int32 array
        out_bits: Target bit width
        saturate: Whether to clamp to valid range
    
    Returns:
        Array of requantized int32 values
    """
    max_val = (1 << out_bits) - 1
    
    result = np.zeros_like(x, dtype=np.int32)
    for i in range(x.size):
        val = int(x.flat[i])
        
        # Clamp
        if saturate:
            val = max(-max_val, min(max_val, val))
        
        # Encode then decode
        code = encode_n(val, out_bits)
        result.flat[i] = decode_n(code, out_bits)
    
    return result
