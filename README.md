# MNISC Compiler

MNISC (Mixed-Neural Inference & Compute) 项目的 OCaml 编译器和 AST 仿真器。

## 项目结构

```
.
├── interfaces/          # 接口定义文件
│   ├── eu_isa.mli      # EU ISA 接口
│   ├── eu_isa.ml       # EU ISA 实现
│   ├── ast_ir.mli      # AST IR 接口
│   └── ast_ir.ml       # AST IR 实现
├── compiler/            # 编译器和仿真器
│   ├── safetensor.ml   # Safetensors 解析器
│   ├── tiling.ml       # Tiling 逻辑
│   ├── lower.ml        # AST -> ISA 转换
│   ├── ast_sim.ml      # AST 仿真器
│   ├── main.ml         # 主程序
│   ├── dune            # dune 构建配置
│   └── dune-project    # dune 项目配置
├── Makefile            # 顶层 Makefile
└── README.md           # 本文件
```

## 构建

需要安装以下依赖：
```bash
opam install yojson bigarray dune
```

构建项目：
```bash
make build
```

## 使用

### 生成随机权重和输入数据

```bash
cd compiler && dune exec ./main.exe -- --gen-weights
```

这将生成：
- `model.safetensors`: 包含 U-Net 权重的 safetensors 文件
- `input.safetensors`: 包含输入数据的 safetensors 文件

### 编译 AST 程序并输出 ISA 指令

```bash
cd compiler && dune exec ./main.exe
```

这将生成：
- `program.bin`: u32 指令流（二进制）
- `program_meta.json`: 程序元数据（JSON）

### 运行 AST 仿真

```bash
cd compiler && dune exec ./main.exe -- --run-sim
```

## MNISC-Q 量化编码

本项目使用 MNISC-Q 非均匀量化编码：

- 2-bit code 解码: 00→-3, 01→-1, 10→+1, 11→+3
- N-bit code: 按 2-bit slice 组合
- decodeN: val = Σ decode2(slice_s) << (2*s)

## U-Net 拓扑

默认实现一个 2-level U-Net：

- 输入: [32, 32, 1], B8 (int8)
- Encoder: 2个下采样阶段
- Bottleneck: [8, 8, 64]
- Decoder: 2个上采样阶段 + skip connections
- 输出: [32, 32, 1], B8

## 指令集

支持的 EU ISA 指令：
- `OPC_CONV3X3`: 3x3 卷积（支持 stride=1/2, pad=0/1）
- `OPC_POOL2D`: 2x2 池化（Max/Avg）
- `OPC_UNPOOL2D`: 2x 上采样（NearestRepeat）
- `OPC_CONCAT_C`: Channel 维度拼接
- `OPC_ACT_QUANT`: 激活 + 重量化
- `OPC_GEMM`: 矩阵乘法（FC 层）

## License

MIT License
