//------------------------------------------------------------------------------
// concat_unit - Channel Concatenation Unit
//------------------------------------------------------------------------------
// Concatenates two tensors along the channel dimension
// Input: Two HWC tensors with same H, W but different C
// Output: HWC tensor with C = C0 + C1
//
// Data flow:
//   1. Read first tensor (C0 channels) from act_in
//   2. Read second tensor (C1 channels) from act_in (same stream, sequential)
//   3. Output concatenated tensor (C0+C1 channels)
//------------------------------------------------------------------------------
module concat_unit #(
  parameter int BUS_W = 128
)(
  input  logic clk,
  input  logic rst_n,
  
  // Configuration
  input  logic [7:0]  cfg_elem_bits,   // Element bits (2/4/8/16/32)
  input  logic [7:0]  cfg_n_inputs,    // Number of inputs (only 2 supported)
  input  logic [15:0] cfg_h,           // Height
  input  logic [15:0] cfg_w,           // Width
  input  logic [15:0] cfg_c0,          // Channels in first input
  input  logic [15:0] cfg_c1,          // Channels in second input
  input  logic [31:0] cfg_bytes0,      // Bytes for first input
  input  logic [31:0] cfg_bytes1,      // Bytes for second input
  
  // Control
  input  logic        start,
  output logic        busy,
  output logic        done,
  
  // Input stream
  input  logic [BUS_W-1:0] act_in_data,
  input  logic             act_in_valid,
  output logic             act_in_ready,
  
  // Output stream
  output logic [BUS_W-1:0] out_data,
  output logic             out_valid,
  input  logic             out_ready
);

  // State machine
  typedef enum logic [2:0] {
    ST_IDLE,
    ST_INPUT0,
    ST_INPUT1,
    ST_DONE
  } state_e;
  
  state_e state;
  
  // Byte counters
  logic [31:0] byte_cnt;
  logic [31:0] bytes_target;
  
  // Position counters (for debugging/tracking)
  logic [15:0] h_cnt;
  logic [15:0] w_cnt;
  logic [15:0] c_cnt;
  
  // Total elements
  logic [31:0] total_elems0;
  logic [31:0] total_elems1;
  logic [31:0] total_elems_out;
  
  assign total_elems0 = cfg_h * cfg_w * cfg_c0;
  assign total_elems1 = cfg_h * cfg_w * cfg_c1;
  assign total_elems_out = total_elems0 + total_elems1;
  
  // State machine
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= ST_IDLE;
      byte_cnt <= 0;
      bytes_target <= 0;
      h_cnt <= 0;
      w_cnt <= 0;
      c_cnt <= 0;
    end else begin
      case (state)
        ST_IDLE: begin
          if (start) begin
            state <= ST_INPUT0;
            byte_cnt <= 0;
            bytes_target <= cfg_bytes0;
            h_cnt <= 0;
            w_cnt <= 0;
            c_cnt <= 0;
          end
        end
        
        ST_INPUT0: begin
          if (act_in_valid && act_in_ready) begin
            byte_cnt <= byte_cnt + (BUS_W / 8);
            
            // Track position (for debugging)
            if (c_cnt + BUS_W / cfg_elem_bits < cfg_c0) begin
              c_cnt <= c_cnt + BUS_W[15:0] / cfg_elem_bits;
            end else begin
              c_cnt <= 0;
              if (w_cnt + 1 < cfg_w) begin
                w_cnt <= w_cnt + 1;
              end else begin
                w_cnt <= 0;
                h_cnt <= h_cnt + 1;
              end
            end
            
            if (byte_cnt + (BUS_W / 8) >= cfg_bytes0) begin
              state <= ST_INPUT1;
              byte_cnt <= 0;
              bytes_target <= cfg_bytes1;
            end
          end
        end
        
        ST_INPUT1: begin
          if (act_in_valid && act_in_ready) begin
            byte_cnt <= byte_cnt + (BUS_W / 8);
            
            if (byte_cnt + (BUS_W / 8) >= cfg_bytes1) begin
              state <= ST_DONE;
            end
          end
        end
        
        ST_DONE: begin
          state <= ST_IDLE;
        end
        
        default: state <= ST_IDLE;
      endcase
    end
  end
  
  // Output generation
  // Simply forward input data to output
  assign out_data = act_in_data;
  assign out_valid = act_in_valid && ((state == ST_INPUT0) || (state == ST_INPUT1));
  
  // Input ready depends on output ready
  assign act_in_ready = out_ready && ((state == ST_INPUT0) || (state == ST_INPUT1));
  
  // Status
  assign busy = (state != ST_IDLE);
  assign done = (state == ST_DONE);

endmodule : concat_unit
