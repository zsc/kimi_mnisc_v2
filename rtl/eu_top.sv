//------------------------------------------------------------------------------
// EU Top Module - MNISC Low-bit Conv3x3 + GEMM Accelerator
//------------------------------------------------------------------------------

module eu_top #(
    parameter int BUS_W = 128,
    parameter int INSN_W = 32,
    parameter int ACC_W = 32
) (
    // Clock and Reset
    input  logic clk,
    input  logic rst_n,
    
    // Instruction Stream Interface
    input  logic                  insn_valid,
    output logic                  insn_ready,
    input  logic [INSN_W-1:0]     insn_data,
    
    // Weight Input Stream Interface
    input  logic                  wgt_in_valid,
    output logic                  wgt_in_ready,
    input  logic [BUS_W-1:0]      wgt_in_data,
    
    // Activation Input Stream Interface
    input  logic                  act_in_valid,
    output logic                  act_in_ready,
    input  logic [BUS_W-1:0]      act_in_data,
    
    // Output Stream Interface
    output logic                  out_valid,
    input  logic                  out_ready,
    output logic [BUS_W-1:0]      out_data,
    
    // Status/Error
    output logic                  done,
    output logic                  error_valid,
    output logic [31:0]           error_code
);

  // Import ISA package contents inside module
  import eu_isa_pkg::*;

  //==========================================================================
  // Internal Signals
  //==========================================================================
  
  // Execution state
  typedef enum logic [2:0] {
    STATE_IDLE,
    STATE_FETCH_INSN,
    STATE_DECODE,
    STATE_EXEC,
    STATE_WAIT_DATA,
    STATE_PROCESSING,
    STATE_OUTPUT,
    STATE_DONE
  } state_e;
  
  state_e state;
  
  // Instruction decode - use logic instead of enum type to avoid forward ref issues
  logic [7:0]                   current_opcode;
  logic [7:0]                   current_flags;
  
  // Byte counters for stream management
  logic [31:0]                  wgt_bytes_counter;
  logic [31:0]                  wgt_bytes_target;
  logic [31:0]                  act_bytes_counter;
  logic [31:0]                  act_bytes_target;
  logic [31:0]                  out_bytes_counter;
  logic [31:0]                  out_bytes_target;
  
  // Current instruction args
  logic [31:0]                  insn_args [0:EU_MAX_ARGS-1];
  logic [$clog2(EU_MAX_ARGS+1)-1:0] arg_count;   // Number of args for current instruction
  logic [$clog2(EU_MAX_ARGS+1)-1:0] arg_idx;     // Current arg index being received
  
  // Instruction interface control
  // insn_ready is high only when EU is IDLE and can accept a new instruction
  logic                         insn_accept;       // Pulse when instruction word is accepted
  
  // Error detection
  logic                         error_pending;
  logic [7:0]                   error_code_pending;

  //==========================================================================
  // Instruction Interface - Simple Ready/Valid Handshake
  //==========================================================================
  
  // insn_ready is high when EU is in IDLE state (can accept new instruction)
  assign insn_ready = (state == STATE_IDLE);
  
  // Accept instruction when valid and ready
  assign insn_accept = insn_valid && insn_ready;

  //==========================================================================
  // State Machine - Sequential Logic
  //==========================================================================
  
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= STATE_IDLE;
      arg_idx <= '0;
      arg_count <= '0;
      wgt_bytes_counter <= '0;
      act_bytes_counter <= '0;
      out_bytes_counter <= '0;
      done <= 1'b0;
      error_valid <= 1'b0;
      error_code <= '0;
      error_pending <= 1'b0;
      error_code_pending <= '0;
      current_opcode <= '0;
      current_flags <= '0;
      
      // Clear instruction args
      for (int i = 0; i < EU_MAX_ARGS; i++) begin
        insn_args[i] <= '0;
      end
      
      // Clear targets
      wgt_bytes_target <= '0;
      act_bytes_target <= '0;
      out_bytes_target <= '0;
      
    end else begin
      // Update error status
      if (error_pending && !error_valid) begin
        error_valid <= 1'b1;
        error_code <= {24'b0, error_code_pending};
      end
      
      case (state)
        STATE_IDLE: begin
          done <= 1'b0;
          
          if (insn_accept) begin
            // Accept instruction word
            insn_args[arg_idx[3:0]] <= insn_data;
            
            if (arg_idx == 0) begin
              // First word: decode opcode and determine arg_count
              current_opcode <= insn_data[7:0];
              current_flags <= insn_data[15:8];
              
              // Determine number of arguments based on opcode
              case (insn_data[7:0])
                OPC_NOP, OPC_END: begin
                  arg_count <= 5'd1;  // Just the opcode word
                end
                OPC_CONV3X3, OPC_GEMM, OPC_POOL2D, 
                OPC_UNPOOL2D, OPC_CONCAT_C, OPC_ACT_QUANT: begin
                  arg_count <= 5'd8;  // 8 words: opcode + 7 params
                end
                default: begin
                  arg_count <= 5'd1;  // Unknown opcode, just accept opcode word
                end
              endcase
            end
            
            // Increment arg_idx
            arg_idx <= arg_idx + 1'b1;
            
            // Check if all args received
            if (arg_idx + 1'b1 >= arg_count && arg_count != 0) begin
              // All arguments received, move to decode
              state <= STATE_DECODE;
            end
            // Otherwise stay in IDLE to receive more args
          end
        end
        
        STATE_FETCH_INSN: begin
          // Not used in new design - fetching happens in IDLE
          // Keep for compatibility, transition to decode
          state <= STATE_DECODE;
        end
        
        STATE_DECODE: begin
          arg_idx <= '0;
          
          case (current_opcode)
            OPC_NOP:  state <= STATE_IDLE;
            OPC_END:  state <= STATE_DONE;
            OPC_CONV3X3,
            OPC_GEMM,
            OPC_POOL2D,
            OPC_UNPOOL2D,
            OPC_CONCAT_C,
            OPC_ACT_QUANT: begin
              state <= STATE_EXEC;
            end
            default: begin
              error_pending <= 1'b1;
              error_code_pending <= ERR_INVALID_OPCODE;
              state <= STATE_DONE;
            end
          endcase
        end
        
        STATE_EXEC: begin
          // Setup byte counters from instruction args
          // insn_args[0]: opcode/flags
          // insn_args[1]: shape config
          // insn_args[2]: tensor dimensions
          // insn_args[3]: tile config
          // insn_args[4]: output dimensions
          wgt_bytes_target <= insn_args[5];  // Weight bytes
          act_bytes_target <= insn_args[6];  // Activation bytes
          out_bytes_target <= insn_args[7];  // Output bytes
          wgt_bytes_counter <= '0;
          act_bytes_counter <= '0;
          out_bytes_counter <= '0;
          state <= STATE_WAIT_DATA;
        end
        
        STATE_WAIT_DATA: begin
          // Wait for all input data
          if ((wgt_bytes_counter >= wgt_bytes_target || wgt_bytes_target == 0) &&
              (act_bytes_counter >= act_bytes_target || act_bytes_target == 0)) begin
            state <= STATE_PROCESSING;
          end
        end
        
        STATE_PROCESSING: begin
          // Processing done when all output sent
          if (out_bytes_counter >= out_bytes_target && out_bytes_target > 0) begin
            state <= STATE_IDLE;
          end
        end
        
        STATE_OUTPUT: begin
          state <= STATE_IDLE;
        end
        
        STATE_DONE: begin
          done <= 1'b1;
          // Stay in DONE until reset or external clear
        end
        
        default: state <= STATE_IDLE;
      endcase
      
      // Byte counters update
      if (state == STATE_WAIT_DATA || state == STATE_PROCESSING) begin
        if (wgt_in_valid && wgt_in_ready) begin
          wgt_bytes_counter <= wgt_bytes_counter + (BUS_W / 8);
        end
        if (act_in_valid && act_in_ready) begin
          act_bytes_counter <= act_bytes_counter + (BUS_W / 8);
        end
        if (out_valid && out_ready) begin
          out_bytes_counter <= out_bytes_counter + (BUS_W / 8);
        end
      end
    end
  end
  
  //==========================================================================
  // Combinational Logic
  //==========================================================================
  
  // Stream Interface Control
  assign wgt_in_ready = (state == STATE_WAIT_DATA) &&
                        (wgt_bytes_counter < wgt_bytes_target);
  
  assign act_in_ready = (state == STATE_WAIT_DATA) &&
                        (act_bytes_counter < act_bytes_target);
  
  assign out_valid = (state == STATE_PROCESSING) &&
                     (out_bytes_counter < out_bytes_target);
  
  // Output data - pass activation through for now (placeholder)
  assign out_data = act_in_data;

  //==========================================================================
  // Assertions (for simulation only)
  //==========================================================================
  
  // synthesis translate_off
  always @(posedge clk) begin
    if (rst_n) begin
      // Debug: trace state and instruction interface
      if (state != STATE_DONE) begin
        $display("[EU_DBG] time=%0t state=%0d insn_ready=%b insn_valid=%b insn_accept=%b arg_idx=%0d arg_count=%0d",
                 $time, state, insn_ready, insn_valid, insn_accept, arg_idx, arg_count);
      end
    end
    if (rst_n && insn_accept) begin
      $display("[EU_DBG] Instruction word accepted: arg_idx=%0d data=%08h", arg_idx, insn_data);
    end
    if (rst_n && error_valid) begin
      $display("[EU_TOP] Error detected: code=%0h at time %0t", error_code, $time);
    end
    if (rst_n && done) begin
      $display("[EU_TOP] Execution completed at time %0t", $time);
    end
  end
  // synthesis translate_on

endmodule : eu_top
