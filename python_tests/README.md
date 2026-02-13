# MNISC Python Test Tools

This directory contains Python tools for the MNISC (Minimal Neural Inference SoC Compiler) project.

## Files

### Core Modules

- **`mnisc_q.py`**: MNISC-Q quantization codec implementation
  - `decode2()`: Decode 2-bit code to signed value
  - `decode_n()`: Decode N-bit MNISC-Q code
  - `encode_n()`: Encode value to N-bit MNISC-Q code
  - `pack_tensor()`: Pack numpy array to MNISC-Q coded bytes
  - `unpack_tensor()`: Unpack bytes to decoded numpy array
  - `requantize()`: Requantize int32 values to N-bit MNISC-Q

### Tools

- **`generate_safetensors.py`**: Generate test data with fake random weights
  ```bash
  python generate_safetensors.py --output-dir ./data --seed 42
  ```

- **`reference_runner.py`**: Python reference implementation of U-Net inference
  ```bash
  python reference_runner.py --model model.safetensors --input input.safetensors --output output.safetensors
  ```

- **`compare_outputs.py`**: Compare two safetensors files
  ```bash
  python compare_outputs.py expected.safetensors actual.safetensors
  ```

- **`run_e2e_test.py`**: End-to-end test runner
  ```bash
  python run_e2e_test.py --output-dir ./test_output --seed 42
  ```

## MNISC-Q Encoding

MNISC-Q is a non-uniform quantization scheme:

- **2-bit code** → value: `00→-3, 01→-1, 10→+1, 11→+3`
- **int4 (4-bit)**: Two 2-bit slices: `val = decode2(slice0) + (decode2(slice1) << 2)`
- **int8 (8-bit)**: Four 2-bit slices: `val = Σ decode2(slice_s) << (2*s)`

## U-Net Topology

The reference implementation follows a 2-level U-Net:

```
Input: [32, 32, 1] (int8)

Encoder:
  conv1a (1→16) → ReLU+quant → conv1b (16→16) → ReLU+quant → skip1
  pool → [16,16,16]
  conv2a (16→32) → ReLU+quant → conv2b (32→32) → ReLU+quant → skip2
  pool → [8,8,32]

Bottleneck:
  conv3a (32→64) → ReLU+quant → conv3b (64→64) → ReLU+quant

Decoder:
  unpool → [16,16,64] → concat(skip2) → [16,16,96]
  conv4a (96→32) → ReLU+quant → conv4b (32→32) → ReLU+quant
  unpool → [32,32,32] → concat(skip1) → [32,32,48]
  conv5a (48→16) → ReLU+quant → conv5b (16→16) → ReLU+quant

Residual: y = x + skip1 (implemented as 1x1 conv)

Output: out_conv (16→1) → quant → [32,32,1]
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage Example

```bash
# 1. Generate test data
python generate_safetensors.py --output-dir ./test_data --seed 42

# 2. Run reference inference
python reference_runner.py \
    --model ./test_data/model.safetensors \
    --input ./test_data/input.safetensors \
    --output ./test_data/output.safetensors

# 3. Compare with expected output (if available)
python compare_outputs.py \
    ./test_data/expected.safetensors \
    ./test_data/output.safetensors
```

## Output Format

All safetensors files follow the standard format:
```
| header_len (8 bytes, uint64, LE) |
| header JSON (UTF-8) |
| padding to 8-byte alignment |
| tensor data |
```

Header JSON format:
```json
{
  "tensor_name": {
    "dtype": "U8",
    "shape": [H, W, C],
    "data_offsets": [start, end]
  }
}
```

## Testing

Run the full test suite:
```bash
python run_e2e_test.py --output-dir ./test_output --keep --verbose
```

The `--keep` flag preserves the output directory for inspection.
