下面给你一套 **可以直接喂给 gemini-cli / codex** 的“分文件 SPEC Markdown 包”（含 ISA / AST 接口文件草案 + RTL / Compiler / Safetensor / OCaml AST Simulator 的规格）。我会尽量把**歧义点都钉死**，并把你上面那份 Conv3x3 low-bit EU 设计融进去，同时补齐：

* ✅ **EU 指令集 ISA**（可执行指令流，配合 ready/valid “DDR stream”）
* ✅ **OCaml AST/IR**（覆盖 U‑Net 常见算子，且可 lower 到 EU 指令）
* ✅ **Tiling 规则**（大 conv/FC 切小块）
* ✅ **Safetensors 生成规范**（fake 随机权重，但层/shape 正确；权重 int4、激活 int8）
* ✅ **OCaml AST 仿真器**（bit‑accurate golden，能与 EU/verilator 输出对分）
* ✅ **Verilator harness 规范**（按 program 驱动流接口，完成 end-to-end）

> 重要修正：你原先 MVP 限制“act_bits>2 与 wgt_bits>2 不同时出现”。但你最终 pipeline 明确要从 **int8 activation × int4 weight** 的 safetensors 开始，因此本套 SPEC 默认 **支持 act_bits 和 wgt_bits 同时 >2**（通过 slice 交叉项计算，见 ISA/RTL/Simulator 里同一套公式）。

---

## 文件树（建议）

```text
spec/
  00_README.md
  01_EU_ISA.md
  02_AST_IR.md
  03_RTL_SPEC.md
  04_COMPILER_SPEC.md
  05_SAFETENSOR_SPEC.md
  06_SIM_AND_DIFF_SPEC.md

interfaces/
  eu_isa.svh
  eu_isa.ml
  eu_isa.mli
  ast_ir.ml
  ast_ir.mli
```

下面每个 code block 就是一份“可落盘”的 markdown / 接口文件内容。

---

```markdown
# spec/00_README.md
# MNISC / EU Low-bit Conv3x3 + FC Accelerator — End-to-End SPEC Bundle

## 1. 目标（要交付什么）
本项目最终要得到一条可验证的端到端链路：

1) `weights.safetensors` + `input.safetensors`（权重 int4 code，激活 int8 code，shape 与 U-Net 模型一致）
2) OCaml 编译器：
   - 解析 safetensors
   - 构建 OCaml AST/IR（`spec/02_AST_IR.md`）
   - 做 tiling，把“大 Conv / 大 FC”切成 EU 可跑的小块
   - lower 到 EU ISA 程序（`spec/01_EU_ISA.md`），产物例如：
     - `program.bin`（u32 指令流）
     - `program.json`（可选 debug trace）
3) OCaml AST 仿真器：
   - 直接解释执行 AST（bit-accurate）
   - 输出各 layer 的 reference tensors（用于对分）
4) SystemVerilog RTL（EU）+ Verilator harness：
   - EU 执行 `program.bin`
   - 外部“DDR”用 ready/valid stream 仿真（权重/激活输入流、输出流）
   - harness 按 program 的元数据把 safetensors 切片打包后喂入 EU
5) 三方对分：
   - (A) EU(verilator) 输出
   - (B) OCaml AST 仿真器输出
   - (C) safetensors reference runner 输出（允许用 Python 参考实现，但语义必须与 AST 一致）
   - 要求 bit-exact match（除非显式启用 tolerance 模式）

## 2. 非目标（本阶段不做）
- 不接 AXI / DMA / DDR controller（用 stream 模拟，未来再替换）
- 不做 Xilinx/Intel 专用 BRAM/DSP IP（只用综合推断）
- 不追求性能极致（可以为正确性增加循环/状态）

## 3. 统一约定（所有模块/工具共享）
### 3.1 数据布局（Tensor 内存线性顺序）
对 2D feature map：shape `[H][W][C]`，C 为 innermost：
`idx = ((y * W) + x) * C + c`

### 3.2 bit packing（存储/传输）
元素 bitwidth `bits ∈ {2,4,8,16,32}`：
- bits=2：1 byte = 4 elems： [1:0]=e0,[3:2]=e1,[5:4]=e2,[7:6]=e3
- bits=4：1 byte = 2 elems： [3:0]=e0,[7:4]=e1
- bits=8：1 byte = 1 elem
- bits=16：2 bytes little-endian
- bits=32：4 bytes little-endian

bus beat：`BUS_W` 位（默认 128b），小端 byte 顺序拼接：
`data[7:0]` 为最低地址 byte。

### 3.3 “MNISC-Q” 非均匀量化编码（核心）
2-bit code 解码为 signed int：
- 00 -> -3
- 01 -> -1
- 10 -> +1
- 11 -> +3

N-bit（N=4/8/16）按 2-bit slice 组合：
- slice_s = bits[2*s+1 : 2*s]
- valN = Σ decode2(slice_s) << (2*s)

注意：这不是标准二补码 int4/int8；safetensors 里存的是“code”，不是数值本身。

### 3.4 卷积输出标定（SHIFT1_EN 默认=1）
所有 2-bit 乘积为奇数；pair 累加为偶数；总和可保证可被 2 整除。
因此默认输出为：
`out = out_full >> 1`（精确，不丢信息）

> 若将来要输出 out_full，可关闭 SHIFT1_EN，但会牺牲 unsigned reduction 技巧。

## 4. 端到端验证
必交：
- 覆盖用例：Conv3x3、Pool、Unpool、Concat、FC（GEMM）、Residual(Add via weights)。
- 三方对分：verilator vs AST sim vs reference runner。

```

---

```systemverilog
// interfaces/eu_isa.svh
package eu_isa_pkg;

  // ---- Global constants ----
  parameter int EU_BUS_W = 128;     // data bus width for streams
  parameter int EU_INSN_W = 32;     // instruction word width
  parameter int EU_MAX_ARGS = 16;   // max u32 args for one instruction (soft limit)

  typedef enum logic [7:0] {
    OPC_NOP        = 8'h00,
    OPC_END        = 8'h01,

    // Data movement / meta (stream-based DDR model)
    OPC_META_TENSOR_DEF = 8'h10, // optional: define tensor handle for debug/runtime
    OPC_META_BAR        = 8'h11, // barrier (debug)

    // EU ops
    OPC_CONV3X3     = 8'h20,
    OPC_POOL2D      = 8'h21,
    OPC_UNPOOL2D    = 8'h22,
    OPC_CONCAT_C    = 8'h23,
    OPC_ACT_QUANT   = 8'h24, // activation + requant
    OPC_GEMM        = 8'h25  // FC / matmul
  } opcode_e;

  // Common: instruction header word format
  // [7:0]  opcode
  // [15:8] flags
  // [31:16] reserved (0)

  // flags bits (shared; per-op may redefine):
  // bit0: CHECK_COUNTS_EN
  // bit1: SHIFT1_EN override (1=enable)
  // bit2: SATURATE_EN
  // bit3: RESERVED

endpackage
```

---

```ocaml
(* interfaces/eu_isa.mli *)
type opcode =
  | Nop
  | End
  | MetaTensorDef
  | MetaBar
  | Conv3x3
  | Pool2D
  | Unpool2D
  | ConcatC
  | ActQuant
  | Gemm

type insn = {
  opcode : opcode;
  flags  : int;      (* 0..255 *)
  args   : int32 list; (* u32 words, little-endian order *)
}

val encode_program_u32 : insn list -> int32 array
```

---

```ocaml
(* interfaces/eu_isa.ml *)
type opcode =
  | Nop | End
  | MetaTensorDef | MetaBar
  | Conv3x3 | Pool2D | Unpool2D | ConcatC | ActQuant | Gemm

type insn = { opcode: opcode; flags:int; args:int32 list }

let opcode_to_u8 = function
  | Nop -> 0x00 | End -> 0x01
  | MetaTensorDef -> 0x10 | MetaBar -> 0x11
  | Conv3x3 -> 0x20 | Pool2D -> 0x21 | Unpool2D -> 0x22
  | ConcatC -> 0x23 | ActQuant -> 0x24 | Gemm -> 0x25

let encode_header ~(opc:int) ~(flags:int) : int32 =
  let opc = opc land 0xFF in
  let flags = flags land 0xFF in
  Int32.of_int (opc lor (flags lsl 8))

let encode_program_u32 (p:insn list) : int32 array =
  let words =
    List.concat_map (fun i ->
      let h = encode_header ~opc:(opcode_to_u8 i.opcode) ~flags:i.flags in
      h :: i.args
    ) p
  in
  Array.of_list words
```

---

```markdown
# spec/01_EU_ISA.md
# EU ISA (Stream-DDR model) — Instruction Set SPEC

> 设计取向：EU 通过 **u32 指令流**执行；数据（weights / activations / outputs）仍通过 ready/valid stream 模拟 DDR。
> Host runtime（verilator harness / 软件驱动）负责按指令的参数从 safetensors 切片并打包，喂入 EU；同时接收 out stream 写回“DDR”。

## 1. 外部接口（EU 顶层）
### 1.1 指令流接口
- `insn_valid`
- `insn_ready`
- `insn_data[31:0]`：u32 little-endian word
- EU 按顺序消费，直到遇到 `OPC_END`。

### 1.2 DDR Stream（与原 conv3x3 spec 一致）
#### Weight 输入流
- `wgt_in_valid/ready/data[BUS_W-1:0]`
#### Activation 输入流
- `act_in_valid/ready/data[BUS_W-1:0]`
#### Output 输出流
- `out_valid/ready/data[BUS_W-1:0]`

> 为简化，本 ISA **不依赖 in_last/out_last** 信号；EU 以指令提供的“期望 byte 计数”结束读写。
> 若实现了 last，也只能用于 debug 校验（CHECK_COUNTS_EN）。

## 2. 指令编码通则
- 指令是变长：`header(u32) + args(u32...)`
- header：
  - bits[7:0] opcode
  - bits[15:8] flags
  - bits[31:16] must be 0

### 2.1 通用 flags（所有 op 共享）
- bit0 `CHECK_COUNTS_EN`：
  - 1：EU 校验 host 实际提供的 bytes 与期望 bytes 一致，否则置 error
  - 0：不校验
- bit1 `SHIFT1_EN`：
  - 1：对所有 MAC 输出做精确 `>>1` 标定（推荐默认=1）
  - 0：输出不右移（需要 signed tree；可选）
- bit2 `SATURATE_EN`：
  - 量化阶段启用 clamp（仅对 `OPC_ACT_QUANT` 有意义）
- bit3..7 reserved

## 3. Tensor / 量化约定（ISA 级）
### 3.1 elem_bits 取值
`elem_bits ∈ {2,4,8,16,32}`

### 3.2 MNISC-Q decode / encode
- decode2：00->-3,01->-1,10->+1,11->+3
- decodeN：按 2-bit slice 组合：
  `val = Σ decode2(slice_s) << (2*s)`

量化（requant）定义（用于 ActQuant）：
- target 表示一个 N-bit code 空间（N=2/4/8/16）
- representable set：所有奇数值 in [-(2^N-1), +(2^N-1)]
- requant(x)：
  1) 如果启用 SATURATE_EN：clamp 到该范围
  2) round_to_nearest_representable_odd：
     - 若 x 为奇数：y=x
     - 若 x 为偶数：y = x + sign(x) * 1 （ties away from 0）
     - 再 clamp 到范围
  3) encodeN(y)：用预计算 LUT（N<=16，表很小）做 value->code 反查

> 这保证 AST sim / reference runner / RTL 行为一致。

## 4. 指令列表与语义
### 4.1 OPC_END
- args：无
- 语义：程序结束。EU 置 done，停止接收后续指令。

### 4.2 OPC_CONV3X3（核心：支持 act_bits 与 wgt_bits 同时 >2）
用途：执行一个 **Conv3x3 tile**，可用于：
- 3x3 conv stride=1/2, pad=0/1
- 1x1 conv：通过让非中心权重=0 实现（编译器负责生成权重）
- residual add：concat 后用 1x1 权重实现通道级加法（见编译器 spec）

#### args（u32 words，按顺序）
1. `mode_bits`:
   - bits[7:0]   act_bits   (2/4/8/16)
   - bits[15:8]  wgt_bits   (2/4/8/16)
   - bits[23:16] stride     (1 or 2)
   - bits[31:24] pad        (0 or 1)  // pad=1 means "same" padding
2. `shape0`:
   - bits[15:0] H_in   (input H)
   - bits[31:16] W_in  (input W)
3. `shape1`:
   - bits[15:0] IC
   - bits[31:16] OC
4. `tile0`:
   - bits[15:0] y0   // tile output start y
   - bits[31:16] x0  // tile output start x
5. `tile1`:
   - bits[15:0] OH_t  // tile output height
   - bits[31:16] OW_t // tile output width
6. `counts_wgt_bytes` (u32)：本 tile 需要从 wgt_in 消费的 bytes
7. `counts_act_bytes` (u32)：本 tile 需要从 act_in 消费的 bytes
8. `counts_out_bytes` (u32)：本 tile 将在 out 输出的 bytes
9. `meta`（可选，推荐保留，EU 可忽略）：
   - bits[15:0] in_tensor_id
   - bits[31:16] out_tensor_id

#### 输入输出张量 layout
- act 输入 layout：HWC packed（C innermost），元素是 act_bits code
- wgt layout：按线性顺序 `[kh][kw][oc][ic]` packed（见 03_RTL_SPEC）
- out layout：HWC packed，元素为 **ACC_W=32** little-endian（默认 raw 输出 int32）
  - 若你要 out 也写回低比特 code：可在 Conv 后接 `OPC_ACT_QUANT`

#### 语义（数学）
令：
- `A(y,x,ic)` 为 decode(act_bits code)
- `W(kh,kw,oc,ic)` 为 decode(wgt_bits code)
- pad：当 pad=1，A 越界视作 0

输出（未量化前）：
`Y_full(oy,ox,oc) = Σ_{ic,kh,kw} A(oy*stride+kh-pad, ox*stride+kw-pad, ic) * W(kh,kw,oc,ic)`

若 SHIFT1_EN=1：
`Y = Y_full >> 1`（精确）

#### 高 bit 交叉项计算（必须支持）
当 act_bits = 2*Sa，wgt_bits=2*Sw：
- A = Σ_{s=0..Sa-1} decode2(a_s) << (2s)
- W = Σ_{g=0..Sw-1} decode2(w_g) << (2g)

则：
`Y_full = Σ_{s,g} ( Conv2b(a_s, w_g) << (2(s+g)) )`

实现允许：
- 通过额外循环（s,g）计算（慢但正确）
- 或通过并行 lane 做 g 并行、s 外循环（推荐实现方式，见 RTL spec）

### 4.3 OPC_POOL2D
用途：2D pooling（默认 2x2 stride=2）。U-Net 下采样。

args：
1. `mode`:
   - bits[7:0] elem_bits (2/4/8/16/32)
   - bits[15:8] pool_kind (0=max, 1=avg)
   - bits[23:16] ksize (only 2 supported in v1)
   - bits[31:24] stride (only 2 supported in v1)
2. `shape`:
   - H_in (u16), W_in (u16)
3. `channels`:
   - C (u16), reserved (u16)
4. `counts_act_bytes` (u32)
5. `counts_out_bytes` (u32)
6. meta ids optional

语义：按 HWC layout，对每个 channel 独立池化。avg 为整数 avg（向零截断）。

### 4.4 OPC_UNPOOL2D
用途：2x upsample（nearest / repeat）。U-Net 上采样。

args：
1. `mode`:
   - bits[7:0] elem_bits
   - bits[15:8] unpool_kind (0=nearest_repeat)
   - bits[23:16] scale (only 2 supported)
2. shape: H_in,W_in
3. channels: C
4. counts_act_bytes
5. counts_out_bytes

语义：nearest_repeat：
`Y(2y+dy,2x+dx,c) = X(y,x,c)` for dy,dx in {0,1}

### 4.5 OPC_CONCAT_C
用途：channel concat（U-Net skip connection）

args：
1. `mode`:
   - bits[7:0] elem_bits
   - bits[15:8] n_inputs (only 2 supported in v1)
2. shape:
   - H (u16), W (u16)
3. channels:
   - C0 (u16), C1 (u16)  // output C=C0+C1
4. counts_act0_bytes (u32) // from act_in stream (first tensor)
5. counts_act1_bytes (u32) // from act_in stream (second tensor) — 注意：仍用同一个 act_in stream，host 顺序发送两段
6. counts_out_bytes (u32)

语义：输出 layout 为 HWC，先拷贝 input0 的 C0，再拷贝 input1 的 C1。

### 4.6 OPC_ACT_QUANT
用途：非线性 + requant。U-Net 常用 ReLU + 量化回 int8。

args：
1. `mode`:
   - bits[7:0] in_bits   (typically 32)
   - bits[15:8] out_bits (2/4/8/16)
   - bits[23:16] act_fn  (0=identity,1=relu)
2. shape: H,W
3. channels: C
4. counts_in_bytes
5. counts_out_bytes

语义：
- 逐元素：x -> act_fn(x) -> requant_to_out_bits(x)

### 4.7 OPC_GEMM（FC / Linear）
用途：矩阵乘（Fully Connected / 1x1 conv 展开）。复用 low-bit MAC 阵列。

布局约定：
- 输入向量/矩阵按 row-major，innermost 为 K（类似 C innermost）
- X: shape [M][K]
- W: shape [N][K]   // 输出 channel N，每个输出有 K 权重
- Y: shape [M][N]

args：
1. `mode`:
   - act_bits (u8), wgt_bits(u8), reserved(u16)
2. shape0:
   - M (u16), K (u16)
3. shape1:
   - N (u16), reserved(u16)
4. counts_wgt_bytes
5. counts_act_bytes
6. counts_out_bytes

语义：
`Y_full[m,n] = Σ_{k} X(m,k) * W(n,k)`
SHIFT1 同 conv。

高 bit 交叉项同 Conv3x3（必须支持）。

## 5. 错误处理（强制）
EU 顶层必须提供：
- `done`（程序结束）
- `error_valid`
- `error_code[31:0]`（至少包含：非法参数/stream underflow/overflow/count mismatch）

```

---

```ocaml
(* interfaces/ast_ir.mli *)

type bits = B2 | B4 | B8 | B16 | B32
type layout = HWC | MK  (* HWC for feature maps; MK for matrices/vectors *)

type tensor_id = string

type tensor = {
  id     : tensor_id;
  bits   : bits;      (* stored as MNISC-Q code except B32 which is int32 *)
  layout : layout;
  shape  : int list;  (* HWC: [H;W;C], MK: [M;K] or [M;N] *)
}

type act_fn = Identity | ReLU
type pool_kind = Max | Avg
type unpool_kind = NearestRepeat

type op =
  | Conv3x3 of {
      input : tensor;
      weight: tensor;          (* [3;3;OC;IC] packed in safetensor as linear *)
      stride: int;             (* 1 or 2 *)
      pad   : int;             (* 0 or 1 *)
      out   : tensor;          (* usually B32 before quant *)
    }
  | Gemm of {
      x     : tensor;          (* MK [M;K] *)
      w     : tensor;          (* NK [N;K] *)
      out   : tensor;          (* MN [M;N], usually B32 *)
    }
  | Pool2D of { input: tensor; kind: pool_kind; out: tensor }
  | Unpool2D of { input: tensor; kind: unpool_kind; out: tensor }
  | ConcatC of { a: tensor; b: tensor; out: tensor } (* H,W equal; C adds *)
  | ActQuant of { input: tensor; fn: act_fn; out_bits: bits; out: tensor } (* out bits != B32 *)
  | Store of { input: tensor }  (* materialize to "DDR" / named output *)
  | Comment of string

type program = op list

val infer_shapes : program -> (tensor_id, tensor) Hashtbl.t
```

---

```ocaml
(* interfaces/ast_ir.ml *)
type bits = B2 | B4 | B8 | B16 | B32
type layout = HWC | MK
type tensor_id = string

type tensor = { id: tensor_id; bits: bits; layout: layout; shape: int list }

type act_fn = Identity | ReLU
type pool_kind = Max | Avg
type unpool_kind = NearestRepeat

type op =
  | Conv3x3 of { input:tensor; weight:tensor; stride:int; pad:int; out:tensor }
  | Gemm of { x:tensor; w:tensor; out:tensor }
  | Pool2D of { input:tensor; kind:pool_kind; out:tensor }
  | Unpool2D of { input:tensor; kind:unpool_kind; out:tensor }
  | ConcatC of { a:tensor; b:tensor; out:tensor }
  | ActQuant of { input:tensor; fn:act_fn; out_bits:bits; out:tensor }
  | Store of { input:tensor }
  | Comment of string

type program = op list

let infer_shapes (_p:program) = Hashtbl.create 32
```

---

```markdown
# spec/02_AST_IR.md
# OCaml AST/IR SPEC — Covers U-Net Operators & Lowering Targets

## 1. AST 设计目标
- 表达 U-Net 常见算子：Conv3x3、Pool、Unpool、Concat、(Residual via weights)、FC/GEMM、激活+量化
- 所有 tensor 都携带：
  - bits（B2/B4/B8/B16/B32）
  - layout（HWC/MK）
  - shape
- 语义必须与 EU ISA / RTL / simulator 一致（MNISC-Q 编码）

## 2. Tensor 约定
### 2.1 Layout = HWC
- shape = [H; W; C]
- 线性索引：((y*W)+x)*C + c
- 存储为 packed code（bits=2/4/8/16）或 int32（bits=32）

### 2.2 Layout = MK
- shape = [M; K] 或 [M; N]（依 op 决定）
- 线性索引：m*K + k（row-major）
- 存储同上

## 3. Op 语义（bit-accurate）
### 3.1 Conv3x3
- 固定 kernel=3
- stride ∈ {1,2}
- pad ∈ {0,1}：
  - pad=0: valid conv，OH=floor((H-3)/stride)+1
  - pad=1: same conv，OH=ceil(H/stride)（但 v1 只支持 stride=1/2 且 pad=1 下 output H=W=H_in/stride 向上取整；编译器需保证 tile 合法）

权重张量 weight 在 AST 层可视为：
- 逻辑 shape：[3;3;OC;IC]
- 物理存储：线性数组按 `[kh][kw][oc][ic]` 顺序 packed（与 EU 一致）

### 3.2 Pool2D（v1 只要求 2x2 stride2）
- kind: Max/Avg
- pad=0
- 输出 shape：[H/2; W/2; C]（要求 H,W 偶数或由编译器 pad/裁剪处理）

### 3.3 Unpool2D（v1 只要求 scale=2）
- NearestRepeat
- 输出 shape：[2H; 2W; C]

### 3.4 ConcatC
- 输入 a,b：H,W 必须一致
- 输出 C = Ca+Cb
- 输出 layout HWC，顺序：先 a 的 channels，再 b 的 channels

### 3.5 Gemm
- x: [M;K], w:[N;K], out:[M;N]
- MNISC-Q decode 后做整数 MAC，SHIFT1 可选（与 ISA flags 一致）

### 3.6 ActQuant
- fn: Identity / ReLU
- 输入 bits 常见为 B32（int32）
- 输出 bits 为 {B2,B4,B8,B16}
- requant 规则完全复用 ISA（nearest representable odd + encode LUT）

## 4. Residual Add 的表达（强制支持）
为了贴合“Residual Add (setting Conv weights)”图：
- AST 可以有显式 Add（可选）
- 但编译器 lowering 必须支持把：
  `y = a + b`
  降为：
  1) ConcatC(a,b) -> t (C=Ca+Cb)
  2) Conv1x1 via Conv3x3:
     - 用 Conv3x3，pad=0/1 取决于 shape
     - 权重只在中心 (kh=1,kw=1) 非零
     - 对于每个 oc（=Ca），令 W(1,1,oc,oc)=1，W(1,1,oc,oc+Ca)=1
     - 其余为 0
  3) 得到 y
这保证 RTL 里无需专用 add 单元。

## 5. 编译器必须实现的静态检查
- bits 只能为 {2,4,8,16,32}
- Conv3x3：IC/OC 与硬件并行度/tiling 约束一致（见 compiler/rtl spec）
- 所有 packed tensor 的总 bits 必须是 8 的倍数（byte 对齐）
- ConcatC：H,W 必须一致
- Pool/Unpool：H,W 符合 2x 变换（或由编译器插入 pad/crop）

```

---

```markdown
# spec/03_RTL_SPEC.md
# RTL SPEC (SystemVerilog) — EU + Low-bit Conv3x3 + GEMM (Vendor-agnostic)

> 本 RTL spec 在你原有 Conv3x3 low-bit spec 基础上扩展：
> 1) 引入 EU ISA decoder / sequencer（执行指令流）
> 2) 支持 act_bits 与 wgt_bits 同时 >2（slice 交叉项）
> 3) 增加 GEMM/FC 模式（复用 muladd2 LUT + unsigned reduction）
> 4) 保留 stream-DDR 外设模型（wgt_in / act_in / out）

---

## 1. 顶层模块
模块名：`eu_top`

### 1.1 参数（综合期常量）
- `BUS_W` default 128
- `INSN_W` default 32
- `IC2_LANES` fixed 16
- `OC2_LANES` fixed 16
- `ACC_W` default 32
- `MAX_H, MAX_W, MAX_IC, MAX_OC`（line buffer / weight buffer 深度上限）
- `SHIFT1_EN_DEFAULT` default 1

### 1.2 端口
时钟复位：
- `clk`
- `rst_n`

指令流：
- `insn_valid, insn_ready, insn_data[31:0]`

DDR streams：
- weight: `wgt_in_valid, wgt_in_ready, wgt_in_data[BUS_W-1:0]`
- activation: `act_in_valid, act_in_ready, act_in_data[BUS_W-1:0]`
- output: `out_valid, out_ready, out_data[BUS_W-1:0]`

状态/错误：
- `done`
- `error_valid`
- `error_code[31:0]`

---

## 2. EU 微体系结构（建议划分）
- `insn_fetch_fifo`：指令 word FIFO
- `eu_sequencer`：解析变长指令，驱动各子模块执行（blocking）
- `stream_counter`：按指令的 counts_*_bytes 计数，生成内部 “end_of_stream”
- `weight_buffer`：按 Conv/Gemm 加载一块权重（或整层权重）
- `feature_line_buffer`：3 行缓存 + padding=0/1 + stride=1/2 窗口发生器
- `conv_core_lowbit`：2-bit LUT + unsigned reduction + slice combine + inter-cycle accumulate
- `gemm_core_lowbit`：K 维 dot-product 的 low-bit MAC（同样的 LUT + reduction + slice combine）
- `pool2d_unit`：2x2 pool
- `unpool2d_unit`：2x nearest repeat
- `concat_unit`：concat 两段输入流到输出流（不需要大缓存；逐元素转发）
- `act_quant_unit`：ReLU + requant + packer
- `output_packer`：把 int32 或 packed code 打包成 BUS_W beat

> v1 强烈建议：每条 EU 指令执行期间，EU 不并发执行下一条指令，避免复杂的 scoreboard。

---

## 3. Conv3x3 Core 关键实现（在你原 spec 基础上修正）
### 3.1 2-bit decode
`decode2(2'b00=-3,01=-1,10=+1,11=+3)`

### 3.2 muladd2_lut（强制）
输入 8bits：a0,w0,a1,w1（各 2bits）
输出 u5：
- p0 = decode2(a0)*decode2(w0)
- p1 = decode2(a1)*decode2(w1)
- pair_sum = p0+p1   // even, range [-18..+18]
- OFFSET=18
- u = (pair_sum + OFFSET) >> 1   // 0..18 unsigned

实现形式：`always_comb case(addr)` 或 ROM 初始化数组，禁止 DSP。

### 3.3 unsigned reduction tree
对每个 oc2_lane、对每个 (kh,kw) 和 ic lane pair：
- 计算所有 u，累加得 sum_u（unsigned）
- N_PAIRS = KH*KW*(IC2_LANES/2) = 3*3*8=72（当一次处理 16 个 ic2_lane）
- 还原：
  sum_s = signed(sum_u) - N_PAIRS*9
- 若 SHIFT1_EN=1：sum_s 即 (dot_products >> 1)

> 如果一次只喂入部分 ic lanes（例如为了支持更多 slice/更小 IC tile），则 N_PAIRS 要按实际 pairs 数更新。

---

## 4. 同时高 bit（act_bits 与 wgt_bits 都 >2）的硬件支持（强制）
### 4.1 Slice 定义
- act_slices = act_bits/2
- wgt_slices = wgt_bits/2

### 4.2 正确公式（必须遵守）
`Y_full = Σ_{s=0..act_slices-1} Σ_{g=0..wgt_slices-1} ( Conv2b(a_s, w_g) << (2*(s+g)) )`

### 4.3 推荐实现方式（不需要双重 (s,g) 全展开）
做法：对 act slice 用外循环，对 weight slice 用 lane 并行：

1) `feature_line_buffer` 增加控制输入：
- `act_slice_sel`（0..act_slices-1）
输出：
- `win_act2[3][3][16]` 只对应该 slice 的 2-bit（每 lane 是一个 IC 通道）

2) `weight_buffer` 输出仍可把所有 wgt slice 按 oc2_lane 分组：
- `wgt2[oc2_lane][3][3][16]`
- oc2_lane 编码 `(g,p)`：g = slice index，p=physical oc lane within slice group

3) `conv_core_lowbit` 一次计算产生所有 oc2_lane 的 partial：
- partial_lane(g,p) = Conv2b(a_s, w_g) >>1

4) `controller` / `conv_core` 内部合并：
- 先对每个 p：`tmp_p = Σ_g ( partial_lane(g,p) << (2*g) )`
- 再乘以 act slice 位权：`tmp_p <<= (2*s)`
- 对同一输出像素与输出通道累加：`acc_p += tmp_p`

这等价于 Σ_{s,g} <<2(s+g)。

---

## 5. feature_line_buffer（加入 pad=1）
### 5.1 存储
存 3 行循环 buffer，每行存 W*IC 个元素（原始 act_bits code）。

### 5.2 Window 生成
输出顺序固定：
`oy -> ox -> ic_grp -> act_slice_sel`

当 pad=1：
- in_y = oy*stride + kh - 1
- in_x = ox*stride + kw - 1
越界返回 0（注意：0 code decode2= -3，不是 0！因此 pad 的“0”必须指 *数值 0*，需要特殊处理）：

**关键点：pad 的“0”是数值 0，不是 MNISC-Q code 00。**
因此 linebuf 在越界时必须直接输出一个特殊 2-bit code，使 decode2 得到 0。
但 decode2 不包含 0！所以必须定义：
- 越界时 act2/wgt2 直接送 “虚拟乘法输入为 0” 的路径。
推荐做法：在 conv_core 前加一个 `act_zero_mask`，越界 lane 直接让乘积为 0（绕过 decode2 LUT）。
实现可选：
- 方法A：在 LUT address 构造时把 a 强制映射到 “0”并在 LUT 内扩展支持 0（addr 增加一位）
- 方法B：保持 LUT 不变，在进入 LUT 前若 mask=1 则输出 u=9（对应 pair_sum=0），并在 offset 还原时抵消
v1 推荐 **方法A**：把 decode 扩展为 3-bit：{ZERO, 00,01,10,11}。

> 如果你坚持 2-bit 不扩展，则 pad=1 无法表达“真 0”，会出错。

因此：**v1 RTL 必须把乘法输入从 2-bit code 扩到 3-bit (zero flag + 2-bit code)**。

---

## 6. weight_buffer
- 存储一条指令范围内的权重（tile）即可，不强制整层。
- weight layout：线性 idx = (((kh*3)+kw)*OC + oc)*IC + ic
- 输出：按 oc2_lane=(g,p) 分 slice。

---

## 7. GEMM Core（FC）
### 7.1 数据流
- 从 `act_in` 读取 X（M*K）
- 从 `wgt_in` 读取 W（N*K）
- 输出 Y（M*N）int32

### 7.2 计算
同 Conv2b dot-product，但没有 KH/KW 维；只在 K 维累加。
- K tile：一次处理 16 个 k（与 IC2_LANES 对齐）
- 对每个输出 lane（oc2_lane）：
  - 用 muladd2_lut 在 16-lane 上做 pair reduction
  - 对 K 维分块 inter-cycle accumulate
- 同样支持 act_bits/wgt_bits 同时高 bit：复用 slice combine 逻辑。

---

## 8. pool/unpool/concat/act_quant
### 8.1 concat
- EU 指令提供 C0,C1,H,W,bits 和 bytes 计数
- concat_unit 按元素顺序：
  - 先读 C0 段，再读 C1 段
  - 输出拼接后的 stream

### 8.2 pool/unpool
- v1 只支持 2x2 stride2 pool；unpool 只支持 2x repeat
- 若输入 bits 为 packed code，需要在 unit 内做 unpack/repack（建议用通用 pack/unpack 库）

### 8.3 act_quant
- 输入通常 int32（B32）
- ReLU + requant_to_bits（LUT value->code）
- 输出 packed code

---

## 9. 验证（RTL 必交 TB）
必须包含：
- 指令流驱动 + stream DDR driver
- golden model（和 spec/06 一致）
- 覆盖用例：
  1) act_bits=8, wgt_bits=4 (关键：同时高 bit)
  2) stride=1,pad=1
  3) stride=2,pad=0
  4) concat + 1x1(conv3x3 center weights) 做 residual add
  5) gemm 小例子
  6) out_ready backpressure

```

---

```markdown
# spec/04_COMPILER_SPEC.md
# OCaml Compiler SPEC — safetensors -> AST -> tiling -> EU ISA program

## 1. 输入输出
输入：
- `model.safetensors`：权重 tensors（int4 code packed）
- `input.safetensors`：输入激活（int8 code packed）
- `model.json`（可选）：网络拓扑描述（若你不想从权重命名推断）

输出：
- `program.bin`：u32 指令流（EU ISA）
- `program_meta.json`：可选，包含每条指令对应的 tensor slice、bytes、shape（给 runtime/harness 用）
- `expected_outputs.safetensors`：可选（AST sim 产物）

## 2. 编译流程（必须实现）
### Pass A：Safetensors 解析
- 解析 header JSON（shape、dtype、offset）
- 支持 dtype：
  - `U8`（存 packed codes）
  - `I32`（bias 或中间 reference，可选）
- 将 tensor data 读入 OCaml bytes/bigarray

### Pass B：构建 AST（front-end）
- 用 `spec/02_AST_IR.md` 的 op 列表构建 U-Net
- v1 允许硬编码一个“参考 U-Net”拓扑（见 spec/05），即：
  - 从 safetensor 中按固定 key 取权重
  - 组装 op list

### Pass C：Shape inference + consistency check
- 检查 concat 维度、pool/unpool 维度、conv stride/pad 合法性
- 检查 bits ∈ {2,4,8,16,32}

### Pass D：Tiling（关键）
硬件约束来自 RTL 参数：
- `MAX_H, MAX_W, MAX_IC, MAX_OC`
- `WBUF_BYTES_MAX`
- `LINEBUF_ROW_BYTES_MAX`（对应 MAX_W*MAX_IC*act_bits/8）
- `OUTBUF_BYTES_MAX`（tile 输出 int32 缓存/或直接流出）

#### D.1 Conv3x3 tiling 规则（建议）
Tile 维度选择优先级：
1) OC 先 tile（因为 weight buffer 和 output lanes 限制）
2) IC tile 用 accumulation（多次 op 或单 op 内循环）
3) H/W tile（保证 line buffer 能装下带 halo 的宽度）

每个 Conv tile 需要：
- 输入区域大小：
  - pad=0: H_in_t = OH_t*stride + 2
  - pad=1: H_in_t = OH_t*stride + 2  // 仍然要读 halo，但越界用 0 mask
  - 类似 W
- act bytes:
  `act_bytes = ceil(H_in_t*W_in_t*IC_t*act_bits / 8)`
- wgt bytes:
  `wgt_bytes = ceil(3*3*IC_t*OC_t*wgt_bits / 8)`
- out bytes（raw int32）：
  `out_bytes = OH_t*OW_t*OC_t*4`

必须满足：
- `wgt_bytes <= WBUF_BYTES_MAX`
- `W_in_t <= MAX_W` 且 `IC_t <= MAX_IC`
- `OC_t <= MAX_OC`
- `act_bytes` 在 runtime 可从“DDR”切片连续喂入（若切片不连续，则 runtime 必须做 gather；v1 推荐 tile 让切片连续）

#### D.2 GEMM tiling
对 K 做 block=16 对齐；对 N 按 OC2_LANES / wgt_slices 的物理 lanes tile；对 M 可 tile。

### Pass E：Lowering 到 EU ISA
对每个 AST op 生成 1..N 条 EU ISA 指令（含 tile 指令序列）。

#### E.1 Conv3x3 lowering
- 每个 tile 一条 `OPC_CONV3X3`
- 若需要 IC accumulation：
  - 方案1：让 RTL 在一条 CONV 指令内部做 ic_grp 循环（推荐）
  - 方案2：拆成多条 CONV（每条处理 IC_t），然后用额外 ADD（但 EU v1 无 ADD），不推荐
因此 v1 规定：**CONV 指令内部必须支持 IC 分组循环**（RTL 需实现）。

#### E.2 Residual Add lowering
将 `y=a+b` 降为：
1) `OPC_CONCAT_C`（读 a,b，输出 t）
2) `OPC_CONV3X3`（用 center 权重实现 1x1 加法）

编译器负责生成该权重 tile，并按 wgt_in stream 送入。

#### E.3 量化 lowering
若下一层需要 B8/B4/B2 输入：
- 在前一层 conv/gemm 输出 B32 后，插入：
  `OPC_ACT_QUANT`

### Pass F：产出 program_meta.json（强烈建议）
对每条 EU 指令，记录：
- opcode, args
- 对 act/wgt/out 对应的 “源 tensor id、tile 范围（y0,x0,ic0,oc0 等）、bytes”
这样 runtime/harness 可以无歧义地切片/打包。

## 3. Runtime/Harness 约定（编译器必须匹配）
- 所有 stream payload 都是严格按 layout 线性顺序 packed
- 若一个 tile 不是连续内存，runtime 允许做 gather，但必须保证 EU 收到的顺序仍然是 tile 的线性顺序
- v1 推荐：H/W tile 仅在整行切分（保持每行连续），channel tile 以对齐边界切分（保证 bit packing 简化）

```

---

```markdown
# spec/05_SAFETENSOR_SPEC.md
# Safetensors SPEC — Random fake weights, correct U-Net shapes (int4 weight, int8 activation)

## 1. 文件列表
- `model.safetensors`：权重（int4 code packed，存为 U8 bytes）
- `input.safetensors`：输入激活（int8 code packed，存为 U8 bytes）
- `ref_output.safetensors`：参考输出（可选：由 Python runner 或 OCaml AST sim 生成）

## 2. MNISC-Q code 存储规则
### 2.1 int4 weight
- 每个 weight 是 4-bit code（两组 2-bit slice）
- 物理存储：packed in bytes（1 byte = 2 weights）
- safetensors dtype：U8
- 解释规则：
  - 低 4 bits 为 weight0 code
  - 高 4 bits 为 weight1 code

### 2.2 int8 activation
- 每个 activation 是 8-bit code（四组 2-bit slice）
- 物理存储：1 byte = 1 activation code
- safetensors dtype：U8

> 注意：U8 只是容器。数值语义由 MNISC-Q decode 决定。

## 3. 参考 U-Net（v1 demo network）
为确保算子覆盖：Conv3x3/Pool/Unpool/Concat/ActQuant/Residual(add via weights)：
采用 2-level U-Net（小尺寸，便于仿真）：

输入：`x0` shape [H=32, W=32, C=1], bits=B8

Encoder:
1) conv1a: Conv3x3 1->16, pad=1, stride=1, out B32
2) relu1a + quant -> B8
3) conv1b: 16->16, pad=1, out B32
4) relu1b + quant -> B8
5) skip1 = output of step4  (for concat)
6) pool1: Pool2D 2x2 stride2 (B8) -> [16,16,16]

7) conv2a: 16->32, pad=1
8) relu+quant -> B8
9) conv2b: 32->32, pad=1
10) relu+quant -> B8
11) skip2 = output of step10
12) pool2 -> [8,8,32]

Bottleneck:
13) conv3a: 32->64, pad=1
14) relu+quant -> B8
15) conv3b: 64->64, pad=1
16) relu+quant -> B8

Decoder:
17) unpool2: upsample -> [16,16,64]
18) concat2: concat(skip2, up) -> [16,16,96]
19) conv4a: 96->32, pad=1
20) relu+quant -> B8
21) conv4b: 32->32, pad=1
22) relu+quant -> B8

23) unpool1 -> [32,32,32]
24) concat1: concat(skip1, up) -> [32,32,48]
25) conv5a: 48->16, pad=1
26) relu+quant -> B8
27) conv5b: 16->16, pad=1
28) relu+quant -> B8

Residual demo（覆盖“Residual Add via weights”）：
29) residual: y = x_skip + x_in
   - 先 concat(x_in, x_in) 得到 [32,32,2]，再用 1x1(conv center) 权重实现 add（或用 skip1 与当前特征做 add）
   - v1 建议：对 final feature 做 residual add：
     a = step28 output [32,32,16]
     b = step4  output [32,32,16]
     y = a + b

Output:
30) out_conv: 1x1 via Conv3x3(center) 16->1, pad=1
31) output bits=B8（可选 quant）

## 4. 权重命名（safetensors keys）
必须使用稳定 key，编译器按 key 取权重。
建议 keys：
- `conv1a.weight` shape [3,3,16,1]  (kh,kw,oc,ic)  packed int4
- `conv1b.weight` shape [3,3,16,16]
- `conv2a.weight` shape [3,3,32,16]
- `conv2b.weight` shape [3,3,32,32]
- `conv3a.weight` shape [3,3,64,32]
- `conv3b.weight` shape [3,3,64,64]
- `conv4a.weight` shape [3,3,32,96]
- `conv4b.weight` shape [3,3,32,32]
- `conv5a.weight` shape [3,3,16,48]
- `conv5b.weight` shape [3,3,16,16]
- `out_conv.weight` shape [3,3,1,16] (only center used)

此外：
- residual add 的 1x1 权重由编译器生成（无需写入 model.safetensors；也可写入 `res_add.weight`）

输入：
- `input.x` shape [32,32,1] bits=B8

## 5. Random fake 权重生成（Python 参考）
- 随机采样 code：
  - int4：0..15
  - int8：0..255
- 直接写 packed bytes
- 生成 `model.safetensors` + `input.safetensors`

要求：
- shape 与 keys 正确
- data length 与 packing 一致（byte 对齐）

```

---

```markdown
# spec/06_SIM_AND_DIFF_SPEC.md
# OCaml AST Simulator + Reference Runner + Verilator Diff SPEC

## 1. OCaml AST 仿真器（必交）
模块：`AstSim`

输入：
- AST program（list of ops）
- tensor store：从 safetensors 加载的 bytes（packed codes）

输出：
- 每个 Store op 的 tensor bytes（packed codes 或 int32）
- 可选：dump 中间层（用于 debug）

### 1.1 核心函数（必须）
- `decode2 : int -> int`（2-bit -> signed）
- `decodeN(bitsN_code) : int`（按 slice 组合）
- `encodeN(value) : code`（反查 LUT）
- `conv3x3_ref`：
  - 输入：HWC packed code
  - 权重：[3,3,OC,IC] packed code
  - 输出：int32
  - 支持 pad=0/1, stride=1/2
  - 支持 act_bits/wgt_bits 同时 >2：用 Σ_{s,g} 公式
- `gemm_ref`：同理
- `pool2d_ref` / `unpool2d_ref` / `concat_ref`
- `act_quant_ref`：ReLU + requant（与 ISA 一致）
- 所有计算使用 int64 累加避免溢出，最终截到 int32（或检查不溢出）

### 1.2 SHIFT1 一致性
仿真器必须接受一个全局开关 SHIFT1_EN（默认 1），与 EU ISA flags 匹配。

## 2. Python reference runner（可选但强烈建议）
目的：独立于 OCaml，再做一份 reference，降低“OCaml sim 自证”的风险。
- 直接用 numpy 实现同样的 decode/conv/quant。
- 读取同一 safetensors。
- 输出 ref_output.safetensors。

## 3. Verilator harness（必交）
### 3.1 输入
- `program.bin`（u32 words）
- `program_meta.json`（用于知道每条指令要喂哪些 tensor slice）
- `model.safetensors` / `input.safetensors`

### 3.2 行为
- reset EU
- 推送指令流：
  - `insn_valid` 按 ready/valid 发送 program words
- 执行期间：
  - 观察 EU 当前等待的 stream（通过内部状态可选导出 debug；或 harness 按 meta 顺序直接喂）
  - 当遇到需要 wgt_in：
    - 按 bytes 计数，从对应 weight 切片打包成 BUS_W beat 推送
  - 当遇到 act_in：
    - 同理喂入 activation
  - 同时接收 out stream：
    - 组装 bytes，写入目标 tensor buffer

### 3.3 对分
对每个 Store 的输出：
- Verilator 输出 bytes vs OCaml AST sim 输出 bytes：
  - 必须 bit-exact
- 若启用 Python runner：
  - 三方一致

### 3.4 Backpressure 测试
harness 必须实现：
- 周期性拉低 `out_ready`（例如每 17 个 cycle 拉低 5 个）
验证 EU 不丢 beat、不乱序。

## 4. 必须输出的日志（便于 debug）
- 每条指令执行开始/结束的 cycle
- 计数器：实际 consumed bytes / expected bytes
- 首个 mismatch 的 tensor name、index、期望/实际值

```

---

## 你现在可以怎么用这套 SPEC

* 把这些 markdown/接口文件保存到仓库（`spec/` + `interfaces/`）。
* 用 gemini-cli/codex 的提示词，让它按这些 SPEC 生成：

  * SystemVerilog RTL（`eu_top` + 子模块）
  * OCaml compiler（safetensors parser + AST builder + tiler + ISA emitter）
  * OCaml AST simulator
  * Python 生成 safetensors + Python reference runner（可选）
  * Verilator harness（C++/Python 都行）

---

## 几个设计决策，override 上面相关部分

1. **pad=1 的“真 0”如何表达**
   我在 RTL spec 里明确指出：MNISC-Q decode2 没有 0，因此 padding 不能靠 code 表达，必须引入 `zero_mask` 或扩展输入编码（推荐扩展 1-bit zero flag）。
   支持。

2. **Conv 指令是否在 RTL 内做 IC 分组循环**
   compiler 实现 tiling，RTL 内无支持。 

3. **输出是否一律先出 int32，再用 ACT_QUANT 回写低比特**
   v1 建议这么做，最稳、最容易对分。
