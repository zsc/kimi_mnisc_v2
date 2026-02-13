//------------------------------------------------------------------------------
// EU ISA Package - MNISC Low-bit Conv3x3 + GEMM Accelerator
//------------------------------------------------------------------------------
package eu_isa_pkg;

  // ---- Global constants ----
  parameter int EU_BUS_W = 128;      // data bus width for streams
  parameter int EU_INSN_W = 32;      // instruction word width
  parameter int EU_MAX_ARGS = 16;    // max u32 args for one instruction (soft limit)
  
  // Default architectural parameters
  parameter int IC2_LANES = 16;      // number of IC 2-bit lanes
  parameter int OC2_LANES = 16;      // number of OC 2-bit lanes  
  parameter int ACC_W = 32;          // accumulator width
  parameter int SHIFT1_EN_DEFAULT = 1;

  // ---- Opcode enumeration ----
  typedef enum logic [7:0] {
    OPC_NOP             = 8'h00,
    OPC_END             = 8'h01,
    
    // Data movement / meta (stream-based DDR model)
    OPC_META_TENSOR_DEF = 8'h10,  // optional: define tensor handle for debug/runtime
    OPC_META_BAR        = 8'h11,  // barrier (debug)
    
    // EU ops
    OPC_CONV3X3         = 8'h20,
    OPC_POOL2D          = 8'h21,
    OPC_UNPOOL2D        = 8'h22,
    OPC_CONCAT_C        = 8'h23,
    OPC_ACT_QUANT       = 8'h24,  // activation + requant
    OPC_GEMM            = 8'h25   // FC / matmul
  } opcode_e;

  // ---- Common flags bits (shared across ops) ----
  // flags[0]: CHECK_COUNTS_EN - verify stream byte counts
  // flags[1]: SHIFT1_EN - enable >>1 calibration on MAC output
  // flags[2]: SATURATE_EN - enable clamp during quantization
  // flags[7:3]: reserved
  
  parameter int FLAG_CHECK_COUNTS = 0;
  parameter int FLAG_SHIFT1_EN    = 1;
  parameter int FLAG_SATURATE_EN  = 2;

  // ---- Error codes ----
  typedef enum logic [7:0] {
    ERR_NONE          = 8'h00,
    ERR_INVALID_OPCODE= 8'h01,
    ERR_INVALID_PARAM = 8'h02,
    ERR_STREAM_UNDERFLOW = 8'h03,
    ERR_STREAM_OVERFLOW  = 8'h04,
    ERR_COUNT_MISMATCH   = 8'h05,
    ERR_UNSUPPORTED_BITS = 8'h06,
    ERR_UNSUPPORTED_STRIDE = 8'h07,
    ERR_UNSUPPORTED_PAD    = 8'h08
  } error_code_e;

  // ---- MNISC-Q decode functions ----
  // 2-bit code -> signed value: 00->-3, 01->-1, 10->+1, 11->+3
  function automatic logic signed [2:0] decode2(input logic [1:0] code);
    case (code)
      2'b00: decode2 = -3;
      2'b01: decode2 = -1;
      2'b10: decode2 = +1;
      2'b11: decode2 = +3;
    endcase
  endfunction

  // ---- 3-bit extended decode (with zero flag) ----
  // For padding support: zero_flag=1 forces output to 0
  // Input: {zero_flag, 2-bit code}
  // Output: signed 3-bit value
  function automatic logic signed [2:0] decode3_ext(input logic [2:0] ext_code);
    logic zero_flag;
    logic [1:0] code;
    zero_flag = ext_code[2];
    code = ext_code[1:0];
    if (zero_flag)
      decode3_ext = 0;
    else
      decode3_ext = decode2(code);
  endfunction

  // ---- Slice count calculation ----
  function automatic int num_slices(input int bits);
    return bits / 2;
  endfunction

  // ---- Bits per element to bytes calculation ----
  function automatic int bits_to_bytes(input int bits);
    return (bits + 7) / 8;
  endfunction

  // ---- Convolution configuration structures ----
  typedef struct packed {
    logic [7:0]  act_bits;
    logic [7:0]  wgt_bits;
    logic [7:0]  stride;
    logic [7:0]  pad;
  } conv_mode_t;

  typedef struct packed {
    logic [15:0] h_in;
    logic [15:0] w_in;
  } conv_shape0_t;

  typedef struct packed {
    logic [15:0] ic;
    logic [15:0] oc;
  } conv_shape1_t;

  typedef struct packed {
    logic [15:0] y0;
    logic [15:0] x0;
  } conv_tile0_t;

  typedef struct packed {
    logic [15:0] oh_t;
    logic [15:0] ow_t;
  } conv_tile1_t;

endpackage : eu_isa_pkg
