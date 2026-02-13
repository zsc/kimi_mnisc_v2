# MNISC - 低比特卷积加速器 (Low-bit Conv3x3 + GEMM Accelerator)

[![SystemVerilog](https://img.shields.io/badge/RTL-SystemVerilog-blue)](rtl/)
[![OCaml](https://img.shields.io/badge/Compiler-OCaml-orange)](compiler/)
[![Python](https://img.shields.io/badge/Tools-Python-green)](python_tests/)

MNISC (Matrix Network Inference & Slice Computation) 是一个端到端的低比特神经网络加速器项目，支持 int8 激活 × int4 权重的推理，包含完整的编译器、RTL 实现和仿真验证环境。

---

## 📋 目录

- [项目概述](#项目概述)
- [核心特性](#核心特性)
- [系统架构](#系统架构)
- [快速开始](#快速开始)
- [目录结构](#目录结构)
- [MNISC-Q 量化方案](#mnisc-q-量化方案)
- [指令集架构 (ISA)](#指令集架构-isa)
- [各组件详细说明](#各组件详细说明)
- [端到端验证流程](#端到端验证流程)
- [开发计划与限制](#开发计划与限制)

---

## 项目概述

MNISC 项目实现了一个完整的 U-Net 推理加速器链路：

1. **量化表示**: 使用 MNISC-Q 非均匀量化编码，支持 2/4/8/16-bit 权重和激活
2. **编译器**: OCaml 实现的完整编译流程（safetensors → AST → Tiling → EU ISA）
3. **RTL 加速器**: SystemVerilog 实现的 EU（Execution Unit），支持 Conv3x3、GEMM、Pool、Unpool、Concat 等算子
4. **仿真验证**: OCaml AST 仿真器 + Python Reference + Verilator RTL 仿真，三方对比验证

**目标网络**: 2-level U-Net（输入 32×32×1，编码器-瓶颈-解码器结构，支持 Skip Connection 和 Residual Add）

---

## 核心特性

### 🔢 MNISC-Q 量化编码

2-bit 编码方案（非标准二补码）：
```
code → value
00   → -3
01   → -1
10   → +1
11   → +3
```

N-bit 数值（N=4/8/16）通过 2-bit slice 组合：
```
val = Σ decode2(slice_s) << (2*s)
```

例如 4-bit (int4)：`val = decode2(bits[1:0]) + (decode2(bits[3:2]) << 2)`

### ⚡ 硬件特性

| 特性 | 参数 |
|------|------|
| 数据总线宽度 | 128-bit |
| 指令宽度 | 32-bit |
| IC 并行度 | 16 lanes (IC2_LANES) |
| OC 并行度 | 16 lanes (OC2_LANES) |
| 累加器位宽 | 32-bit |
| 支持卷积核 | 3×3 (stride 1/2, pad 0/1) |
| 支持矩阵乘 | GEMM (FC/Linear) |

### 🔄 支持的算子

- **Conv3x3**: 支持 int8 激活 × int4 权重，多 slice 交叉项计算
- **GEMM**: 矩阵乘法，复用 Conv MAC 阵列
- **Pool2D**: 2×2 Max/Average Pooling，stride=2
- **Unpool2D**: 2× Nearest Neighbor 上采样
- **ConcatC**: Channel 维度拼接（Skip Connection）
- **ActQuant**: ReLU + Requantization（量化回 int8/int4/int2）

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         软件栈                                    │
├─────────────────────────────────────────────────────────────────┤
│  Python Tools                                                    │
│  ├── generate_safetensors.py  ← 生成测试权重和输入                │
│  ├── reference_runner.py      ← Python 参考实现                   │
│  └── compare_outputs.py       ← 输出对比工具                      │
├─────────────────────────────────────────────────────────────────┤
│  OCaml Compiler                                                  │
│  ├── Safetensor parser        ← 解析 safetensors 格式             │
│  ├── AST/IR builder           ← 构建 U-Net 计算图                 │
│  ├── Tiling                   ← 大算子切分策略                    │
│  ├── Lowering                 ← AST → EU ISA 指令                 │
│  └── AST Simulator            ← bit-accurate 参考仿真             │
└─────────────────────────────────────────────────────────────────┘
                              ↓ program.bin
┌─────────────────────────────────────────────────────────────────┐
│                         硬件层                                    │
├─────────────────────────────────────────────────────────────────┤
│  EU (Execution Unit)                                             │
│  ├── Instruction Decoder      ← 指令译码                         │
│  ├── Feature Line Buffer      ← 3行特征缓存 + padding            │
│  ├── Weight Buffer            ← 权重缓存                         │
│  ├── Conv3x3 Core             ← 2-bit MAC 阵列                   │
│  ├── GEMM Core                ← 矩阵乘单元                       │
│  ├── Pool/Unpool/Concat Units ← 辅助算子单元                     │
│  └── ActQuant Unit            ← 激活+量化                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 快速开始

### 环境要求

- **OCaml**: 4.14+ (opam, dune, yojson)
- **Verilator**: 5.0+ (用于 RTL 仿真)
- **Python**: 3.8+ (numpy, safetensors)
- **C++ 编译器**: 支持 C++17

### 安装依赖

```bash
# OCaml 依赖
opam install yojson -y

# Python 依赖
pip install numpy safetensors

# macOS Verilator
brew install verilator

# Linux Verilator
sudo apt-get install verilator
```

### 构建项目

```bash
# 构建所有组件
make build

# 或分别构建
cd compiler && dune build          # OCaml 编译器
cd rtl && make                     # Verilator 仿真模型
```

### 运行端到端测试

```bash
# 1. 生成测试数据
python python_tests/generate_safetensors.py

# 2. 运行 Python Reference
python python_tests/reference_runner.py \
    --model python_tests/model.safetensors \
    --input python_tests/input.safetensors \
    --output python_tests/ref_output.safetensors

# 3. OCaml 编译器生成指令
./compiler/_build/default/main.exe
# 生成: program.bin, program_meta.json

# 4. RTL 仿真
cd rtl && make run
```

---

## 目录结构

```
.
├── SPEC.md                    # 详细规范文档
├── README.md                  # 本文件
├── Makefile                   # 顶层构建脚本
├── .gitignore                 # Git 忽略配置
│
├── interfaces/                # OCaml 接口定义
│   ├── eu_isa.mli/ml         # EU ISA 类型和编码
│   └── ast_ir.mli/ml         # AST/IR 类型定义
│
├── compiler/                  # OCaml 编译器和仿真器
│   ├── dune-project          # Dune 项目配置
│   ├── dune                  # 构建配置
│   ├── safetensor.ml         # Safetensors 解析器
│   ├── ast_ir.ml             # AST/IR 实现
│   ├── tiling.ml             # Tiling 策略
│   ├── lower.ml              # AST → ISA Lowering
│   ├── ast_sim.ml            # AST 仿真器
│   ├── main.ml               # 主程序入口
│   └── test_mnisc.ml         # 单元测试
│
├── rtl/                       # SystemVerilog RTL
│   ├── Makefile              # RTL 构建脚本
│   ├── eu_isa_pkg.sv         # ISA 包定义
│   ├── eu_top.sv             # EU 顶层模块
│   ├── muladd2_lut.sv        # 2-bit 乘法 LUT
│   ├── feature_line_buffer.sv# 特征行缓存
│   ├── weight_buffer.sv      # 权重缓存
│   ├── conv_core_lowbit.sv   # Conv3x3 核心
│   ├── gemm_core_lowbit.sv   # GEMM 核心
│   ├── pool2d_unit.sv        # Pooling 单元
│   ├── unpool2d_unit.sv      # Unpooling 单元
│   ├── concat_unit.sv        # Concat 单元
│   ├── act_quant_unit.sv     # 激活+量化单元
│   ├── eu_sequencer.sv       # 指令序列器
│   └── tb_eu_top.cpp         # Verilator Testbench
│
└── python_tests/              # Python 工具
    ├── requirements.txt      # Python 依赖
    ├── mnisc_q.py            # MNISC-Q 编解码
    ├── generate_safetensors.py # 生成测试数据
    ├── reference_runner.py   # Python 参考实现
    ├── compare_outputs.py    # 输出对比
    └── run_e2e_test.py       # E2E 测试脚本
```

---

## MNISC-Q 量化方案

### 编码原理

MNISC-Q 是一种非均匀量化编码，通过 2-bit slice 的移位累加构建数值：

```python
# 2-bit 解码
def decode2(code):
    mapping = {0b00: -3, 0b01: -1, 0b10: +1, 0b11: +3}
    return mapping[code]

# N-bit 解码 (N=4/8/16)
def decode_n(code, n_bits):
    result = 0
    for s in range(n_bits // 2):
        slice_val = (code >> (2*s)) & 0b11
        result += decode2(slice_val) << (2*s)
    return result

# 4-bit 示例
decode_n(0b0001, 4)  # decode2(01) + decode2(00)<<2 = -1 + (-3)<<2 = -13
```

### 可表示数值范围

| 位宽 | 最小值 | 最大值 | 可表示数值 |
|------|--------|--------|-----------|
| 2-bit | -3 | +3 | -3, -1, +1, +3 |
| 4-bit | -45 | +45 | 奇数序列 |
| 8-bit | -765 | +765 | 奇数序列 |

### 存储格式

- **权重 (int4)**: 1 byte = 2 个权重，`[low_4bits, high_4bits]`
- **激活 (int8)**: 1 byte = 1 个激活
- **数据布局**: HWC (Height × Width × Channel)，C 为 innermost

---

## 指令集架构 (ISA)

### 指令格式

变长指令：`header(u32) + args(u32...)`

**Header 格式**:
```
[7:0]   - opcode
[15:8]  - flags
[31:16] - reserved (0)
```

**Flags**:
- bit0: CHECK_COUNTS_EN (校验字节计数)
- bit1: SHIFT1_EN (MAC 结果右移 1 位，默认 1)
- bit2: SATURATE_EN (量化阶段启用 clamp)

### 操作码列表

| Opcode | 值 | 说明 |
|--------|-----|------|
| NOP | 0x00 | 空操作 |
| END | 0x01 | 程序结束 |
| META_TENSOR_DEF | 0x10 | 张量定义（调试） |
| META_BAR | 0x11 | Barrier（调试） |
| CONV3X3 | 0x20 | 3×3 卷积 |
| POOL2D | 0x21 | 2D 池化 |
| UNPOOL2D | 0x22 | 2D 上采样 |
| CONCAT_C | 0x23 | Channel 拼接 |
| ACT_QUANT | 0x24 | 激活+量化 |
| GEMM | 0x25 | 矩阵乘法 |

### CONV3X3 指令参数

```c
args[0]: mode_bits
  [7:0]   act_bits   (2/4/8/16)
  [15:8]  wgt_bits   (2/4/8/16)
  [23:16] stride     (1 or 2)
  [31:24] pad        (0 or 1)
  
args[1]: shape0
  [15:0]  H_in
  [31:16] W_in
  
args[2]: shape1
  [15:0]  IC
  [31:16] OC
  
args[3]: tile0
  [15:0]  y0 (输出起始 y)
  [31:16] x0 (输出起始 x)
  
args[4]: tile1
  [15:0]  OH_t (tile 输出高度)
  [31:16] OW_t (tile 输出宽度)
  
args[5]: counts_wgt_bytes
args[6]: counts_act_bytes
args[7]: counts_out_bytes
args[8]: meta (可选) tensor ids
```

---

## 各组件详细说明

### 1. OCaml 编译器

#### Safetensors 解析 (`safetensor.ml`)

解析 HuggingFace safetensors 格式：
- 读取 header JSON（shape、dtype、data_offsets）
- 支持 dtype: U8（packed codes）、I32（bias）
- 按 tensor key 读取数据

#### AST/IR (`ast_ir.ml`)

计算图表示：
```ocaml
type op =
  | Conv3x3 of { input:tensor; weight:tensor; stride:int; pad:int; out:tensor }
  | Gemm of { x:tensor; w:tensor; out:tensor }
  | Pool2D of { input:tensor; kind:pool_kind; out:tensor }
  | Unpool2D of { input:tensor; kind:unpool_kind; out:tensor }
  | ConcatC of { a:tensor; b:tensor; out:tensor }
  | ActQuant of { input:tensor; fn:act_fn; out_bits:bits; out:tensor }
  | Store of { input:tensor }
```

#### Tiling (`tiling.ml`)

硬件约束：
- `MAX_IC = 16`, `MAX_OC = 16`
- `WBUF_BYTES_MAX` 权重缓存限制
- `LINEBUF_ROW_BYTES_MAX` 行缓存限制

Tiling 策略：
1. OC 先 tile（权重缓存限制）
2. IC tile 用 accumulation
3. H/W tile（保证行缓存能装下带 halo 的宽度）

#### Lowering (`lower.ml`)

AST → EU ISA：
- Conv3x3 → OPC_CONV3X3
- Pool2D → OPC_POOL2D
- Residual Add → ConcatC + Conv1x1（center weight）

### 2. RTL 模块

#### eu_top (顶层)

接口：
```systemverilog
// Instruction
input  logic                  insn_valid,
output logic                  insn_ready,
input  logic [INSN_W-1:0]     insn_data,

// Weight Stream
input  logic                  wgt_in_valid,
output logic                  wgt_in_ready,
input  logic [BUS_W-1:0]      wgt_in_data,

// Activation Stream
input  logic                  act_in_valid,
output logic                  act_in_ready,
input  logic [BUS_W-1:0]      act_in_data,

// Output Stream
output logic                  out_valid,
input  logic                  out_ready,
output logic [BUS_W-1:0]      out_data
```

状态机：
- `IDLE` → 等待指令
- `FETCH_INSN` → 接收指令参数
- `DECODE` → 译码
- `EXEC` → 配置执行单元
- `WAIT_DATA` → 等待输入数据流
- `PROCESSING` → 执行计算
- `OUTPUT` → 输出结果

#### conv_core_lowbit (卷积核心)

关键特性：
- **muladd2_lut**: 2-bit × 2-bit 乘法 LUT（纯逻辑，无 DSP）
- **Unsigned Reduction Tree**: pair_sum 转换后无符号累加
- **Slice Combine**: 支持多 bit 交叉项 `Σ Conv2b(a_s, w_g) << 2(s+g)`

计算公式：
```
pair_sum = decode2(a) * decode2(w)   // range [-18, +18]
OFFSET = 18
u = (pair_sum + OFFSET) >> 1         // unsigned 0..18

sum_u = Σ u (unsigned tree)
sum_s = sum_u - N_PAIRS * 18         // restore signed
if SHIFT1_EN: result = sum_s         // already >> 1
```

#### feature_line_buffer (特征行缓存)

- 3行循环缓冲区
- 支持 pad=0/1, stride=1/2
- 输出 3×3 窗口，带 zero_mask（padding 位置输出 0）

### 3. Python 工具

#### reference_runner.py

完整的 U-Net 推理实现，包括：
- Encoder: conv1a/b → pool1 → conv2a/b → pool2
- Bottleneck: conv3a/b
- Decoder: unpool2 → concat2 → conv4a/b → unpool1 → concat1 → conv5a/b
- Residual: skip1 + final feature

#### mnisc_q.py

MNISC-Q 编解码核心：
- `decode2/decode_n`: code → value
- `encode_n`: value → code（nearest odd）
- `pack_tensor/unpack_tensor`: 与 numpy 数组互转

---

## 端到端验证流程

### 三方对分验证

```
                    ┌──────────────────┐
                    │  model.safetensors│
                    │  input.safetensors│
                    └────────┬─────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐   ┌────────────────┐   ┌────────────────┐
│ Python Runner │   │ OCaml AST Sim  │   │ Verilator (RTL)│
│  (Reference)  │   │   (Golden)     │   │  (Hardware)    │
└───────┬───────┘   └───────┬────────┘   └───────┬────────┘
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐   ┌────────────────┐   ┌────────────────┐
│  ref_output   │   │ ast_output     │   │ rtl_output     │
└───────────────┘   └────────────────┘   └────────────────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ compare_outputs  │
                    │  (bit-exact match)│
                    └──────────────────┘
```

### 验证命令

```bash
# 1. 生成测试数据
python python_tests/generate_safetensors.py

# 2. Python Reference
python python_tests/reference_runner.py \
    --model python_tests/model.safetensors \
    --input python_tests/input.safetensors \
    --output python_tests/ref_output.safetensors

# 3. OCaml AST 仿真
./compiler/_build/default/main.exe --run-sim

# 4. RTL 仿真
cd rtl && make run

# 5. 对比输出
python python_tests/compare_outputs.py \
    --ref python_tests/ref_output.safetensors \
    --actual rtl/verilator_output.safetensors
```

---

## 开发计划与限制

### 已完成功能 ✅

- [x] MNISC-Q 量化编解码（2/4/8-bit）
- [x] OCaml 编译器完整流程
- [x] OCaml AST 仿真器（所有算子）
- [x] Python Reference 实现
- [x] RTL 基础架构 + 指令接口
- [x] Verilator Harness

### 进行中/待完善 🚧

- [ ] RTL 计算核心完整逻辑（目前是骨架）
- [ ] Verilator harness 流控优化（insn_ready 握手）
- [ ] 完整 E2E 测试通过
- [ ] Backpressure 测试（周期性拉低 ready）

### 已知限制

1. **Padding 零值**: 使用 3-bit 扩展格式 `{zero_flag, code}` 表示数值 0
2. **Residual Add**: 通过 ConcatC + 1x1 Conv（center weight）实现，无需专用加法器
3. **指令 FIFO**: 当前 EU 一次处理一条指令，需要 testbench 配合流控

---

## 参考文献

- [MNISC-Q 量化方案](SPEC.md#33-mnisc-q-非均匀量化编码)
- [EU ISA 规范](SPEC.md#eu-isa-stream-ddr-model--instruction-set-spec)
- [AST/IR 规范](SPEC.md#ocaml-astir-spec)
- [RTL 规范](SPEC.md#rtl-spec-systemverilog)

---

## 许可证

MIT License - 详见 [LICENSE](LICENSE)

---

## 贡献者

- 项目基于 SPEC.md 规范实现
- 使用 Verilator 进行 RTL 仿真
- OCaml 编译器使用 dune 构建系统
