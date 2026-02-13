//------------------------------------------------------------------------------
// eu_sequencer - EU Instruction Sequencer
//------------------------------------------------------------------------------
// Decodes EU ISA instructions and controls execution units
// Supports:
//   - OPC_NOP, OPC_END
//   - OPC_CONV3X3, OPC_GEMM
//   - OPC_POOL2D, OPC_UNPOOL2D
//   - OPC_CONCAT_C, OPC_ACT_QUANT
//------------------------------------------------------------------------------
module eu_sequencer #(
  parameter int BUS_W = 128,
  parameter int INSN_W = 32,
  parameter int IC2_LANES = 16,
  parameter int OC2_LANES = 16,
  parameter int ACC_W = 32
)(
  input  logic clk,
  input  logic rst_n,
  
  // Instruction stream
  input  logic         insn_valid,
  output logic         insn_ready,
  input  logic [31:0]  insn_data,
  
  // Status
  output logic         done,
  output logic         error_valid,
  output logic [31:0]  error_code,
  
  // Control outputs to execution units
  // Conv3x3
  output logic         conv_start,
  input  logic         conv_busy,
  input  logic         conv_done,
  output logic [7:0]   conv_cfg_act_bits,
  output logic [7:0]   conv_cfg_wgt_bits,
  output logic [7:0]   conv_cfg_stride,
  output logic         conv_cfg_pad,
  output logic [15:0]  conv_cfg_h_in,
  output logic [15:0]  conv_cfg_w_in,
  output logic [15:0]  conv_cfg_ic,
  output logic [15:0]  conv_cfg_oc,
  output logic [31:0]  conv_cfg_wgt_bytes,
  output logic [31:0]  conv_cfg_act_bytes,
  output logic [31:0]  conv_cfg_out_bytes,
  
  // GEMM
  output logic         gemm_start,
  input  logic         gemm_busy,
  input  logic         gemm_done,
  output logic [15:0]  gemm_cfg_m,
  output logic [15:0]  gemm_cfg_k,
  output logic [15:0]  gemm_cfg_n,
  output logic [7:0]   gemm_cfg_act_bits,
  output logic [7:0]   gemm_cfg_wgt_bits,
  
  // Pool2D
  output logic         pool_start,
  input  logic         pool_busy,
  input  logic         pool_done,
  output logic [7:0]   pool_cfg_elem_bits,
  output logic         pool_cfg_kind,
  
  // Unpool2D
  output logic         unpool_start,
  input  logic         unpool_busy,
  input  logic         unpool_done,
  output logic [7:0]   unpool_cfg_elem_bits,
  
  // Concat
  output logic         concat_start,
  input  logic         concat_busy,
  input  logic         concat_done,
  output logic [7:0]   concat_cfg_elem_bits,
  output logic [31:0]  concat_cfg_bytes0,
  output logic [31:0]  concat_cfg_bytes1,
  
  // ActQuant
  output logic         quant_start,
  input  logic         quant_busy,
  input  logic         quant_done,
  output logic [7:0]   quant_cfg_in_bits,
  output logic [7:0]   quant_cfg_out_bits,
  output logic [7:0]   quant_cfg_act_fn
);

  import eu_isa_pkg::*;
  
  // Instruction fetch state
  typedef enum logic [3:0] {
    SEQ_IDLE,
    SEQ_FETCH_HDR,
    SEQ_FETCH_ARGS,
    SEQ_DECODE,
    SEQ_EXEC,
    SEQ_WAIT_DONE,
    SEQ_ERROR
  } seq_state_e;
  
  seq_state_e seq_state;
  
  // Instruction buffer
  logic [31:0] insn_buf [0:EU_MAX_ARGS-1];
  logic [4:0]  insn_arg_cnt;
  logic [4:0]  insn_num_args;
  
  // Current instruction
  opcode_e  curr_opcode;
  logic [7:0] curr_flags;
  
  // Argument count per opcode
  function automatic int get_num_args(input opcode_e opc);
    case (opc)
      OPC_NOP:         return 0;
      OPC_END:         return 0;
      OPC_META_TENSOR_DEF: return 4;
      OPC_META_BAR:    return 0;
      OPC_CONV3X3:     return 8;
      OPC_POOL2D:      return 5;
      OPC_UNPOOL2D:    return 5;
      OPC_CONCAT_C:    return 5;
      OPC_ACT_QUANT:   return 4;
      OPC_GEMM:        return 5;
      default:         return 0;
    endcase
  endfunction
  
  // Extract fields from instruction
  assign curr_opcode = opcode_e'(insn_buf[0][7:0]);
  assign curr_flags = insn_buf[0][15:8];
  
  // State machine
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      seq_state <= SEQ_IDLE;
      insn_arg_cnt <= 0;
      insn_num_args <= 0;
      done <= 0;
      error_valid <= 0;
      error_code <= 0;
      
      // Clear control signals
      conv_start <= 0;
      gemm_start <= 0;
      pool_start <= 0;
      unpool_start <= 0;
      concat_start <= 0;
      quant_start <= 0;
      
    end else begin
      // Default: clear single-cycle pulses
      conv_start <= 0;
      gemm_start <= 0;
      pool_start <= 0;
      unpool_start <= 0;
      concat_start <= 0;
      quant_start <= 0;
      done <= 0;
      error_valid <= 0;
      
      case (seq_state)
        SEQ_IDLE: begin
          if (insn_valid) begin
            seq_state <= SEQ_FETCH_HDR;
            insn_buf[0] <= insn_data;
            insn_arg_cnt <= 0;
          end
        end
        
        SEQ_FETCH_HDR: begin
          // Decode opcode and determine number of arguments
          insn_num_args <= get_num_args(curr_opcode);
          if (get_num_args(curr_opcode) > 0) begin
            seq_state <= SEQ_FETCH_ARGS;
          end else begin
            seq_state <= SEQ_DECODE;
          end
        end
        
        SEQ_FETCH_ARGS: begin
          if (insn_valid) begin
            insn_arg_cnt <= insn_arg_cnt + 1;
            insn_buf[insn_arg_cnt + 1] <= insn_data;
            if (insn_arg_cnt + 1 >= insn_num_args) begin
              seq_state <= SEQ_DECODE;
            end
          end
        end
        
        SEQ_DECODE: begin
          // Setup execution based on opcode
          case (curr_opcode)
            OPC_NOP: begin
              seq_state <= SEQ_IDLE;
            end
            
            OPC_END: begin
              done <= 1;
              seq_state <= SEQ_IDLE;
            end
            
            OPC_CONV3X3: begin
              // Parse arguments
              conv_cfg_act_bits <= insn_buf[1][7:0];
              conv_cfg_wgt_bits <= insn_buf[1][15:8];
              conv_cfg_stride <= insn_buf[1][23:16];
              conv_cfg_pad <= insn_buf[1][24];
              conv_cfg_h_in <= insn_buf[2][15:0];
              conv_cfg_w_in <= insn_buf[2][31:16];
              conv_cfg_ic <= insn_buf[3][15:0];
              conv_cfg_oc <= insn_buf[3][31:16];
              // Skip tile args for now
              conv_cfg_wgt_bytes <= insn_buf[6];
              conv_cfg_act_bytes <= insn_buf[7];
              conv_cfg_out_bytes <= insn_buf[8];
              
              conv_start <= 1;
              seq_state <= SEQ_WAIT_DONE;
            end
            
            OPC_GEMM: begin
              gemm_cfg_act_bits <= insn_buf[1][7:0];
              gemm_cfg_wgt_bits <= insn_buf[1][15:8];
              gemm_cfg_m <= insn_buf[2][15:0];
              gemm_cfg_k <= insn_buf[2][31:16];
              gemm_cfg_n <= insn_buf[3][15:0];
              
              gemm_start <= 1;
              seq_state <= SEQ_WAIT_DONE;
            end
            
            OPC_POOL2D: begin
              pool_cfg_elem_bits <= insn_buf[1][7:0];
              pool_cfg_kind <= insn_buf[1][15];
              
              pool_start <= 1;
              seq_state <= SEQ_WAIT_DONE;
            end
            
            OPC_UNPOOL2D: begin
              unpool_cfg_elem_bits <= insn_buf[1][7:0];
              
              unpool_start <= 1;
              seq_state <= SEQ_WAIT_DONE;
            end
            
            OPC_CONCAT_C: begin
              concat_cfg_elem_bits <= insn_buf[1][7:0];
              concat_cfg_bytes0 <= insn_buf[4];
              concat_cfg_bytes1 <= insn_buf[5];
              
              concat_start <= 1;
              seq_state <= SEQ_WAIT_DONE;
            end
            
            OPC_ACT_QUANT: begin
              quant_cfg_in_bits <= insn_buf[1][7:0];
              quant_cfg_out_bits <= insn_buf[1][15:8];
              quant_cfg_act_fn <= insn_buf[1][23:16];
              
              quant_start <= 1;
              seq_state <= SEQ_WAIT_DONE;
            end
            
            default: begin
              error_valid <= 1;
              error_code <= {24'd0, ERR_INVALID_OPCODE};
              seq_state <= SEQ_ERROR;
            end
          endcase
        end
        
        SEQ_WAIT_DONE: begin
          // Wait for current operation to complete
          case (curr_opcode)
            OPC_CONV3X3:  if (conv_done) seq_state <= SEQ_IDLE;
            OPC_GEMM:     if (gemm_done) seq_state <= SEQ_IDLE;
            OPC_POOL2D:   if (pool_done) seq_state <= SEQ_IDLE;
            OPC_UNPOOL2D: if (unpool_done) seq_state <= SEQ_IDLE;
            OPC_CONCAT_C: if (concat_done) seq_state <= SEQ_IDLE;
            OPC_ACT_QUANT: if (quant_done) seq_state <= SEQ_IDLE;
            default: seq_state <= SEQ_IDLE;
          endcase
        end
        
        SEQ_ERROR: begin
          // Stay in error state until reset
          error_valid <= 1;
        end
        
        default: seq_state <= SEQ_IDLE;
      endcase
    end
  end
  
  // Instruction ready
  assign insn_ready = (seq_state == SEQ_IDLE) || (seq_state == SEQ_FETCH_ARGS);

endmodule : eu_sequencer
