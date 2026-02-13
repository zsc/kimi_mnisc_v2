#!/usr/bin/env python3
"""
MNISC Python Reference Runner

Implements U-Net inference with numpy, matching OCaml AST simulator semantics.
Uses MNISC-Q quantization for weights (int4) and activations (int8).
"""

import os
import sys
import json
import struct
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable

from mnisc_q import decode2, decode_n, encode_n, unpack_tensor, pack_tensor, requantize


# ============================================================================
# Core Operations
# ============================================================================

def conv3x3_ref(act: np.ndarray, wgt: np.ndarray,
                stride: int = 1, pad: int = 1, shift1: bool = True) -> np.ndarray:
    """
    Conv3x3 reference implementation.
    
    Args:
        act: Input activation [H, W, IC] (int32 decoded values)
        wgt: Weight [3, 3, OC, IC] (int32 decoded values)
        stride: Stride (1 or 2)
        pad: Padding (0 or 1)
        shift1: Whether to apply >>1 shift (default True per spec)
    
    Returns:
        Output [OH, OW, OC] (int32)
    """
    H, W, IC = act.shape
    KH, KW, OC, IC_w = wgt.shape
    assert KH == 3 and KW == 3, "Only 3x3 kernel supported"
    assert IC == IC_w, f"Channel mismatch: act={IC}, wgt={IC_w}"
    
    # Calculate output dimensions
    if pad == 0:
        OH = (H - KH) // stride + 1
        OW = (W - KW) // stride + 1
    else:  # pad == 1
        OH = (H + 2 * pad - KH) // stride + 1
        OW = (W + 2 * pad - KW) // stride + 1
    
    output = np.zeros((OH, OW, OC), dtype=np.int64)
    
    # Pad input if needed
    if pad > 0:
        act_padded = np.pad(act, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0)
    else:
        act_padded = act
    
    # Convolution
    for oy in range(OH):
        for ox in range(OW):
            for oc in range(OC):
                acc = 0
                for kh in range(KH):
                    for kw in range(KW):
                        iy = oy * stride + kh
                        ix = ox * stride + kw
                        for ic in range(IC):
                            a = int(act_padded[iy, ix, ic])
                            w = int(wgt[kh, kw, oc, ic])
                            acc += a * w
                output[oy, ox, oc] = acc
    
    # Apply shift1 if enabled
    if shift1:
        output = output >> 1
    
    return output.astype(np.int32)


def pool2d_ref(x: np.ndarray, kind: str = 'max') -> np.ndarray:
    """
    2x2 stride2 pool.
    
    Args:
        x: Input [H, W, C]
        kind: 'max' or 'avg'
    
    Returns:
        Output [H/2, W/2, C]
    """
    H, W, C = x.shape
    assert H % 2 == 0 and W % 2 == 0, "H and W must be even"
    
    OH, OW = H // 2, W // 2
    output = np.zeros((OH, OW, C), dtype=np.int32)
    
    for oy in range(OH):
        for ox in range(OW):
            for c in range(C):
                # 2x2 window
                vals = [
                    x[oy*2,   ox*2,   c],
                    x[oy*2,   ox*2+1, c],
                    x[oy*2+1, ox*2,   c],
                    x[oy*2+1, ox*2+1, c]
                ]
                if kind == 'max':
                    output[oy, ox, c] = max(vals)
                else:  # avg
                    output[oy, ox, c] = sum(vals) // 4
    
    return output


def unpool2d_ref(x: np.ndarray) -> np.ndarray:
    """
    2x nearest repeat upsample.
    
    Args:
        x: Input [H, W, C]
    
    Returns:
        Output [2*H, 2*W, C]
    """
    H, W, C = x.shape
    output = np.zeros((2*H, 2*W, C), dtype=np.int32)
    
    for y in range(H):
        for x_idx in range(W):
            for c in range(C):
                val = x[y, x_idx, c]
                output[2*y,   2*x_idx,   c] = val
                output[2*y,   2*x_idx+1, c] = val
                output[2*y+1, 2*x_idx,   c] = val
                output[2*y+1, 2*x_idx+1, c] = val
    
    return output


def concat_c_ref(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Channel concat.
    
    Args:
        a: Input [H, W, C0]
        b: Input [H, W, C1]
    
    Returns:
        Output [H, W, C0+C1]
    """
    H, W, C0 = a.shape
    H2, W2, C1 = b.shape
    assert H == H2 and W == W2, "Spatial dims must match"
    
    return np.concatenate([a, b], axis=2)


def act_quant_ref(x: np.ndarray, fn: str = 'relu', out_bits: int = 8) -> np.ndarray:
    """
    Activation + quantize: ReLU + requant.
    
    Args:
        x: Input [H, W, C] (typically int32)
        fn: 'relu' or 'identity'
        out_bits: Target bit width (2, 4, 8, 16)
    
    Returns:
        Output [H, W, C] (requantized values)
    """
    # Apply activation
    if fn == 'relu':
        x = np.maximum(x, 0)
    # identity: no change
    
    # Requantize
    return requantize(x, out_bits, saturate=True)


def residual_add_via_conv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Residual add implemented as 1x1 conv (center weight=1).
    
    This concatenates a and b, then uses a 1x1 conv to add them.
    
    Args:
        a: Input [H, W, C]
        b: Input [H, W, C] (same shape as a)
    
    Returns:
        Output [H, W, C] = a + b
    """
    H, W, C = a.shape
    
    # Concatenate along channel: [H, W, 2C]
    concat = concat_c_ref(a, b)
    
    # Create 1x1 weights that add the two halves
    # Weight shape: [3, 3, C, 2C] - only center (kh=1, kw=1) is used
    wgt = np.zeros((3, 3, C, 2*C), dtype=np.int32)
    for oc in range(C):
        # From first half (a)
        wgt[1, 1, oc, oc] = 1
        # From second half (b)
        wgt[1, 1, oc, oc + C] = 1
    
    # Conv3x3 with pad=1 (but effectively 1x1 since only center is non-zero)
    output = conv3x3_ref(concat, wgt, stride=1, pad=1, shift1=True)
    
    return output


# ============================================================================
# U-Net Model
# ============================================================================

class UNetModel:
    """U-Net model implementation for MNISC."""
    
    def __init__(self, weights: Dict[str, np.ndarray]):
        """
        Initialize with loaded weights.
        
        Args:
            weights: Dict of weight tensors (decoded int32 values)
        """
        self.weights = weights
    
    def forward(self, x: np.ndarray, return_all: bool = False) -> Dict[str, np.ndarray]:
        """
        Run full U-Net forward pass.
        
        Args:
            x: Input [32, 32, 1] (int32 decoded values)
            return_all: If True, return all intermediate activations
        
        Returns:
            Dict with 'output' and optionally intermediate activations
        """
        results = {'input': x}
        
        # ========== Encoder ==========
        # 1) conv1a: Conv3x3 1->16, pad=1, stride=1
        x = conv3x3_ref(x, self.weights['conv1a.weight'], stride=1, pad=1)
        x = act_quant_ref(x, fn='relu', out_bits=8)
        
        # 3) conv1b: 16->16, pad=1
        x = conv3x3_ref(x, self.weights['conv1b.weight'], stride=1, pad=1)
        x = act_quant_ref(x, fn='relu', out_bits=8)
        
        # 5) skip1 = output
        skip1 = x.copy()
        results['skip1'] = skip1
        
        # 6) pool1: 2x2 stride2 -> [16,16,16]
        x = pool2d_ref(x, kind='max')
        
        # 7) conv2a: 16->32, pad=1
        x = conv3x3_ref(x, self.weights['conv2a.weight'], stride=1, pad=1)
        x = act_quant_ref(x, fn='relu', out_bits=8)
        
        # 9) conv2b: 32->32, pad=1
        x = conv3x3_ref(x, self.weights['conv2b.weight'], stride=1, pad=1)
        x = act_quant_ref(x, fn='relu', out_bits=8)
        
        # 11) skip2 = output
        skip2 = x.copy()
        results['skip2'] = skip2
        
        # 12) pool2 -> [8,8,32]
        x = pool2d_ref(x, kind='max')
        
        # ========== Bottleneck ==========
        # 13) conv3a: 32->64, pad=1
        x = conv3x3_ref(x, self.weights['conv3a.weight'], stride=1, pad=1)
        x = act_quant_ref(x, fn='relu', out_bits=8)
        
        # 15) conv3b: 64->64, pad=1
        x = conv3x3_ref(x, self.weights['conv3b.weight'], stride=1, pad=1)
        x = act_quant_ref(x, fn='relu', out_bits=8)
        
        # ========== Decoder ==========
        # 17) unpool2: upsample -> [16,16,64]
        x = unpool2d_ref(x)
        
        # 18) concat2: concat(skip2, up) -> [16,16,96]
        x = concat_c_ref(skip2, x)
        
        # 19) conv4a: 96->32, pad=1
        x = conv3x3_ref(x, self.weights['conv4a.weight'], stride=1, pad=1)
        x = act_quant_ref(x, fn='relu', out_bits=8)
        
        # 21) conv4b: 32->32, pad=1
        x = conv3x3_ref(x, self.weights['conv4b.weight'], stride=1, pad=1)
        x = act_quant_ref(x, fn='relu', out_bits=8)
        
        # 23) unpool1 -> [32,32,32]
        x = unpool2d_ref(x)
        
        # 24) concat1: concat(skip1, up) -> [32,32,48]
        x = concat_c_ref(skip1, x)
        
        # 25) conv5a: 48->16, pad=1
        x = conv3x3_ref(x, self.weights['conv5a.weight'], stride=1, pad=1)
        x = act_quant_ref(x, fn='relu', out_bits=8)
        
        # 27) conv5b: 16->16, pad=1
        x = conv3x3_ref(x, self.weights['conv5b.weight'], stride=1, pad=1)
        x = act_quant_ref(x, fn='relu', out_bits=8)
        
        # ========== Residual Add ==========
        # 29) residual: y = x + skip1 (both [32,32,16])
        x = residual_add_via_conv(x, skip1)
        x = act_quant_ref(x, fn='relu', out_bits=8)
        
        # ========== Output ==========
        # 30) out_conv: 1x1 via Conv3x3(center) 16->1, pad=1
        x = conv3x3_ref(x, self.weights['out_conv.weight'], stride=1, pad=1)
        # Note: out_conv only uses center weights effectively
        x = act_quant_ref(x, fn='identity', out_bits=8)
        
        results['output'] = x
        return results


# ============================================================================
# Safetensors I/O
# ============================================================================

def load_safetensors(path: str) -> Dict[str, Tuple[np.ndarray, List[int], str]]:
    """
    Load safetensors file.
    
    Returns:
        Dict of {name: (array, shape, dtype)}
    """
    with open(path, 'rb') as f:
        data = f.read()
    
    # Read header length
    header_len = struct.unpack('<Q', data[:8])[0]
    
    # Read and parse header
    header_bytes = data[8:8+header_len]
    header = json.loads(header_bytes.decode('utf-8'))
    
    # Calculate data start offset
    header_end = 8 + header_len
    padding = (8 - (header_end % 8)) % 8
    data_start = header_end + padding
    
    # Extract tensors
    tensors = {}
    for name, info in header.items():
        shape = info['shape']
        dtype = info['dtype']
        offsets = info['data_offsets']
        
        tensor_data = data[data_start + offsets[0]:data_start + offsets[1]]
        
        if dtype == 'I32':
            arr = np.frombuffer(tensor_data, dtype=np.int32).reshape(shape)
        elif dtype == 'U8':
            # Determine bits from tensor name or context
            # Default: weights are int4, input is int8
            if 'weight' in name:
                bits = 4
            else:
                bits = 8
            arr = unpack_tensor(tensor_data, bits, shape)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
        
        tensors[name] = (arr, shape, dtype)
    
    return tensors


def save_safetensors(tensors: Dict[str, Tuple[np.ndarray, List[int], str]], path: str):
    """
    Save tensors to safetensors file.
    
    Args:
        tensors: Dict of {name: (array, shape, dtype)}
        dtype: 'I32' for int32, 'U8' for packed codes
    """
    # Build header
    header = {}
    offset = 0
    data_parts = []
    
    for name, (arr, shape, dtype) in tensors.items():
        if dtype == 'I32':
            tensor_bytes = arr.astype(np.int32).tobytes()
        elif dtype == 'U8':
            # Pack based on bit width
            # Infer bits from array values range or use 8 as default
            tensor_bytes = pack_tensor(arr, 8)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
        
        data_len = len(tensor_bytes)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + data_len]
        }
        offset += data_len
        data_parts.append(tensor_bytes)
    
    # Serialize header
    header_json = json.dumps(header, separators=(',', ':'))
    header_bytes = header_json.encode('utf-8')
    
    header_len = len(header_bytes)
    header_len_bytes = struct.pack('<Q', header_len)
    
    # Padding
    padding_needed = (8 - (len(header_bytes) % 8)) % 8
    padding = b'\x00' * padding_needed
    
    # Write file
    with open(path, 'wb') as f:
        f.write(header_len_bytes)
        f.write(header_bytes)
        f.write(padding)
        for part in data_parts:
            f.write(part)


def load_model_weights(path: str) -> Dict[str, np.ndarray]:
    """Load and decode model weights from safetensors."""
    tensors = load_safetensors(path)
    weights = {}
    for name, (arr, shape, dtype) in tensors.items():
        weights[name] = arr
        print(f"  Loaded {name}: shape={shape}, range=[{arr.min()}, {arr.max()}]")
    return weights


def load_input_tensor(path: str) -> np.ndarray:
    """Load and decode input tensor from safetensors."""
    tensors = load_safetensors(path)
    if 'input.x' in tensors:
        arr, shape, dtype = tensors['input.x']
        print(f"  Loaded input.x: shape={shape}, range=[{arr.min()}, {arr.max()}]")
        return arr
    else:
        # Return first tensor
        name = list(tensors.keys())[0]
        arr, shape, dtype = tensors[name]
        print(f"  Loaded {name}: shape={shape}, range=[{arr.min()}, {arr.max()}]")
        return arr


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='MNISC Reference Runner')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model.safetensors')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input.safetensors')
    parser.add_argument('--output', type=str, default='ref_output.safetensors',
                        help='Output path for results')
    parser.add_argument('--dump-intermediates', action='store_true',
                        help='Dump all intermediate activations')
    
    args = parser.parse_args()
    
    print("Loading model weights...")
    weights = load_model_weights(args.model)
    
    print("\nLoading input tensor...")
    x = load_input_tensor(args.input)
    
    print("\nRunning inference...")
    model = UNetModel(weights)
    results = model.forward(x, return_all=args.dump_intermediates)
    
    print("\nOutput statistics:")
    for name, arr in results.items():
        print(f"  {name}: shape={arr.shape}, range=[{arr.min()}, {arr.max()}], mean={arr.mean():.2f}")
    
    print(f"\nSaving output to {args.output}...")
    output_tensor = results['output']
    output_dict = {
        'output': (output_tensor, list(output_tensor.shape), 'I32')
    }
    
    if args.dump_intermediates:
        for name, arr in results.items():
            if name != 'output':
                output_dict[name] = (arr, list(arr.shape), 'I32')
    
    save_safetensors(output_dict, args.output)
    print("Done!")


if __name__ == '__main__':
    main()
