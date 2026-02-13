#!/usr/bin/env python3
"""
Compare two safetensors files and report differences.
"""

import sys
import json
import struct
import argparse
import numpy as np
from typing import Dict, List, Tuple

from mnisc_q import unpack_tensor


def load_safetensors_raw(path: str) -> Dict[str, Tuple[bytes, List[int], str]]:
    """Load safetensors file returning raw bytes."""
    with open(path, 'rb') as f:
        data = f.read()
    
    header_len = struct.unpack('<Q', data[:8])[0]
    header_bytes = data[8:8+header_len]
    header = json.loads(header_bytes.decode('utf-8'))
    
    header_end = 8 + header_len
    padding = (8 - (header_end % 8)) % 8
    data_start = header_end + padding
    
    tensors = {}
    for name, info in header.items():
        shape = info['shape']
        dtype = info['dtype']
        offsets = info['data_offsets']
        tensor_data = data[data_start + offsets[0]:data_start + offsets[1]]
        tensors[name] = (tensor_data, shape, dtype, info['data_offsets'])
    
    return tensors


def load_safetensors(path: str) -> Dict[str, np.ndarray]:
    """Load safetensors file returning decoded arrays."""
    raw = load_safetensors_raw(path)
    tensors = {}
    for name, (data, shape, dtype, offsets) in raw.items():
        if dtype == 'I32':
            arr = np.frombuffer(data, dtype=np.int32).copy()
        elif dtype == 'I64':
            arr = np.frombuffer(data, dtype=np.int64).copy()
        elif dtype == 'F32':
            arr = np.frombuffer(data, dtype=np.float32).copy()
        elif dtype == 'U8':
            # Determine bit width from context
            if 'weight' in name:
                bits = 4
            else:
                bits = 8
            arr = unpack_tensor(data, bits, shape)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
        tensors[name] = arr.reshape(shape)
    return tensors


def compare_tensors(arr1: np.ndarray, arr2: np.ndarray, name: str, tolerance: int = 0) -> bool:
    """
    Compare two tensors and report differences.
    
    Returns:
        True if match, False otherwise
    """
    if arr1.shape != arr2.shape:
        print(f"  [MISMATCH] {name}: Shape mismatch {arr1.shape} vs {arr2.shape}")
        return False
    
    if arr1.dtype != arr2.dtype:
        print(f"  [WARNING] {name}: Dtype mismatch {arr1.dtype} vs {arr2.dtype}")
    
    diff = np.abs(arr1.astype(np.int64) - arr2.astype(np.int64))
    max_diff = diff.max()
    
    if max_diff <= tolerance:
        print(f"  [OK] {name}: shape={arr1.shape}, max_diff={max_diff}")
        return True
    
    # Find first mismatch
    mismatch_idx = np.unravel_index(np.argmax(diff), diff.shape)
    
    print(f"  [MISMATCH] {name}:")
    print(f"    Shape: {arr1.shape}")
    print(f"    Max difference: {max_diff}")
    print(f"    First mismatch at index: {mismatch_idx}")
    print(f"    Expected: {arr1[mismatch_idx]}")
    print(f"    Got:      {arr2[mismatch_idx]}")
    
    # Count mismatches
    mismatch_count = np.sum(diff > tolerance)
    total_elems = arr1.size
    print(f"    Mismatch count: {mismatch_count}/{total_elems} ({100*mismatch_count/total_elems:.2f}%)")
    
    return False


def main():
    parser = argparse.ArgumentParser(description='Compare two safetensors files')
    parser.add_argument('file1', help='First file (expected)')
    parser.add_argument('file2', help='Second file (actual)')
    parser.add_argument('--tolerance', type=int, default=0,
                        help='Tolerance for numerical differences (default: 0 for bit-exact)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed statistics for all tensors')
    
    args = parser.parse_args()
    
    print(f"Loading {args.file1}...")
    tensors1 = load_safetensors(args.file1)
    
    print(f"Loading {args.file2}...")
    tensors2 = load_safetensors(args.file2)
    
    print("\nComparing tensors...")
    
    all_match = True
    checked = set()
    
    for name in tensors1:
        if name not in tensors2:
            print(f"  [MISSING] {name}: present in file1 but not in file2")
            all_match = False
            continue
        
        match = compare_tensors(tensors1[name], tensors2[name], name, args.tolerance)
        if not match:
            all_match = False
        checked.add(name)
    
    for name in tensors2:
        if name not in checked:
            print(f"  [EXTRA] {name}: present in file2 but not in file1")
    
    print()
    if all_match:
        print("[PASS] All tensors match!")
        return 0
    else:
        print("[FAIL] Some tensors do not match.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
