#!/usr/bin/env python3
"""
MNISC Safetensors Generator

Generates fake random weights and inputs with correct shapes and U-Net topology.
Weights use int4 MNISC-Q code, activations use int8 MNISC-Q code.
"""

import os
import json
import struct
import argparse
import numpy as np
from typing import Dict, List, Tuple, Any

# Import MNISC-Q codec
from mnisc_q import encode_n, pack_tensor


# U-Net weight specifications
# Format: (name, shape, wgt_bits)
# Shape: [kh, kw, oc, ic] for conv weights
WEIGHT_SPECS = [
    ("conv1a.weight", [3, 3, 16, 1], 4),
    ("conv1b.weight", [3, 3, 16, 16], 4),
    ("conv2a.weight", [3, 3, 32, 16], 4),
    ("conv2b.weight", [3, 3, 32, 32], 4),
    ("conv3a.weight", [3, 3, 64, 32], 4),
    ("conv3b.weight", [3, 3, 64, 64], 4),
    ("conv4a.weight", [3, 3, 32, 96], 4),
    ("conv4b.weight", [3, 3, 32, 32], 4),
    ("conv5a.weight", [3, 3, 16, 48], 4),
    ("conv5b.weight", [3, 3, 16, 16], 4),
    ("out_conv.weight", [3, 3, 1, 16], 4),
]

# Input specification
INPUT_NAME = "input.x"
INPUT_SHAPE = [32, 32, 1]  # [H, W, C]
INPUT_BITS = 8


def generate_random_codes(shape: List[int], bits: int, seed: int = None) -> bytes:
    """
    Generate random MNISC-Q codes and pack them into bytes.
    
    Args:
        shape: Tensor shape
        bits: Bit width per element
        seed: Random seed for reproducibility
    
    Returns:
        Packed bytes
    """
    if seed is not None:
        np.random.seed(seed)
    
    total_elems = np.prod(shape)
    
    # Generate random codes in valid range
    max_code = (1 << bits) - 1
    codes = np.random.randint(0, max_code + 1, size=total_elems, dtype=np.int32)
    
    # Pack to bytes
    return pack_tensor(codes, bits)


def generate_deterministic_codes(shape: List[int], bits: int, offset: int = 0) -> bytes:
    """
    Generate deterministic MNISC-Q codes for testing.
    Uses a simple pattern based on index.
    
    Args:
        shape: Tensor shape
        bits: Bit width per element
        offset: Offset added to pattern
    
    Returns:
        Packed bytes
    """
    total_elems = np.prod(shape)
    max_code = (1 << bits) - 1
    
    # Generate pattern: cycling through codes
    codes = np.array([(i + offset) % (max_code + 1) for i in range(total_elems)], dtype=np.int32)
    
    return pack_tensor(codes, bits)


def build_safetensors(tensors: Dict[str, Tuple[bytes, List[int], str]]) -> bytes:
    """
    Build safetensors file from tensors.
    
    Args:
        tensors: Dict of {name: (data_bytes, shape, dtype)}
                 dtype: "U8" for packed codes, "I32" for int32
    
    Returns:
        Safetensors file bytes
    """
    # Build header
    header = {}
    offset = 0
    
    for name, (data, shape, dtype) in tensors.items():
        data_len = len(data)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + data_len]
        }
        offset += data_len
    
    # Serialize header to JSON
    header_json = json.dumps(header, separators=(',', ':'))
    header_bytes = header_json.encode('utf-8')
    
    # Header length (8 bytes, uint64, little-endian)
    header_len = len(header_bytes)
    header_len_bytes = struct.pack('<Q', header_len)
    
    # Padding to 8-byte alignment
    padding_needed = (8 - (header_len % 8)) % 8
    padding = b'\x00' * padding_needed
    
    # Concatenate all parts
    result = header_len_bytes + header_bytes + padding
    
    for name, (data, shape, dtype) in tensors.items():
        result += data
    
    return result


def generate_model_weights(random: bool = True, seed: int = 42) -> Dict[str, Tuple[bytes, List[int], str]]:
    """
    Generate all U-Net model weights.
    
    Args:
        random: If True, use random values; otherwise use deterministic pattern
        seed: Random seed
    
    Returns:
        Dict suitable for build_safetensors
    """
    tensors = {}
    
    for name, shape, bits in WEIGHT_SPECS:
        if random:
            data = generate_random_codes(shape, bits, seed=seed)
        else:
            # Use hash of name as offset for deterministic variation
            offset = hash(name) % 1000
            data = generate_deterministic_codes(shape, bits, offset=offset)
        
        tensors[name] = (data, shape, "U8")
        seed += 1  # Vary seed for each tensor
    
    return tensors


def generate_input_tensor(random: bool = True, seed: int = 100) -> Dict[str, Tuple[bytes, List[int], str]]:
    """
    Generate input tensor.
    
    Args:
        random: If True, use random values; otherwise use deterministic pattern
        seed: Random seed
    
    Returns:
        Dict suitable for build_safetensors
    """
    tensors = {}
    
    if random:
        data = generate_random_codes(INPUT_SHAPE, INPUT_BITS, seed=seed)
    else:
        data = generate_deterministic_codes(INPUT_SHAPE, INPUT_BITS, offset=0)
    
    tensors[INPUT_NAME] = (data, INPUT_SHAPE, "U8")
    
    return tensors


def main():
    parser = argparse.ArgumentParser(description='Generate MNISC Safetensors files')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Output directory for generated files')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--deterministic', action='store_true',
                        help='Use deterministic pattern instead of random')
    parser.add_argument('--model-only', action='store_true',
                        help='Only generate model weights')
    parser.add_argument('--input-only', action='store_true',
                        help='Only generate input tensor')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    use_random = not args.deterministic
    
    # Generate model weights
    if not args.input_only:
        print("Generating model weights...")
        model_tensors = generate_model_weights(random=use_random, seed=args.seed)
        model_path = os.path.join(args.output_dir, 'model.safetensors')
        model_data = build_safetensors(model_tensors)
        with open(model_path, 'wb') as f:
            f.write(model_data)
        print(f"  Saved: {model_path}")
        print(f"  Tensors: {list(model_tensors.keys())}")
        total_bytes = sum(len(d) for d, s, t in model_tensors.values())
        print(f"  Total bytes: {total_bytes}")
    
    # Generate input tensor
    if not args.model_only:
        print("Generating input tensor...")
        input_tensors = generate_input_tensor(random=use_random, seed=args.seed + 100)
        input_path = os.path.join(args.output_dir, 'input.safetensors')
        input_data = build_safetensors(input_tensors)
        with open(input_path, 'wb') as f:
            f.write(input_data)
        print(f"  Saved: {input_path}")
        print(f"  Shape: {INPUT_SHAPE}, bits: {INPUT_BITS}")
        total_bytes = sum(len(d) for d, s, t in input_tensors.values())
        print(f"  Total bytes: {total_bytes}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
