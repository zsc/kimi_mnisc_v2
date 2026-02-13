//------------------------------------------------------------------------------
// act_quant_unit - Activation + Quantization Unit
//------------------------------------------------------------------------------
// Features:
//   - Activation function: Identity or ReLU
//   - Requantization to lower bits (2/4/8/16)
//   - Saturation support
//   - MNISC-Q encoding
//------------------------------------------------------------------------------
module act_quant_unit #(
  parameter int BUS_W = 128,
  parameter int ACC_W = 32
)(
  input  logic clk,
  input  logic rst_n,
  
  // Configuration
  input  logic [7:0]  cfg_in_bits,    // Input bits (typically 32)
  input  logic [7:0]  cfg_out_bits,   // Output bits (2/4/8/16)
  input  logic [7:0]  cfg_act_fn,     // 0=Identity, 1=ReLU
  input  logic        cfg_saturate_en,// Enable saturation
  input  logic [15:0] cfg_h,          // Height
  input  logic [15:0] cfg_w,          // Width
  input  logic [15:0] cfg_c,          // Channels
  
  // Control
  input  logic        start,
  output logic        busy,
  output logic        done,
  
  // Input stream (int32)
  input  logic [BUS_W-1:0] act_in_data,
  input  logic             act_in_valid,
  output logic             act_in_ready,
  
  // Output stream (packed code)
  output logic [BUS_W-1:0] out_data,
  output logic             out_valid,
  input  logic             out_ready
);

  // State machine
  typedef enum logic [2:0] {
    ST_IDLE,
    ST_PROCESS,
    ST_DONE
  } state_e;
  
  state_e state;
  
  // Element counters
  logic [31:0] elem_cnt;
  logic [31:0] total_elems;
  
  // Calculate total elements
  assign total_elems = cfg_h * cfg_w * cfg_c;
  
  // Elements per beat
  logic [7:0] in_elems_per_beat;
  logic [7:0] out_elems_per_beat;
  
  assign in_elems_per_beat = BUS_W / cfg_in_bits;
  assign out_elems_per_beat = BUS_W / cfg_out_bits;
  
  // Requantization range
  logic signed [ACC_W-1:0] min_val;
  logic signed [ACC_W-1:0] max_val;
  
  // MNISC-Q representable range for out_bits
  // For N bits: odd values in [-(2^N-1), +(2^N-1)]
  always_comb begin
    case (cfg_out_bits)
      2: begin min_val = -3; max_val = 3; end
      4: begin min_val = -15; max_val = 15; end
      8: begin min_val = -255; max_val = 255; end
      16: begin min_val = -32767; max_val = 32767; end
      default: begin min_val = -3; max_val = 3; end
    endcase
  end
  
  // Element processing function
  // Input: int32 value
  // Output: N-bit MNISC-Q code
  function automatic logic [15:0] quantize_elem(
    input logic signed [ACC_W-1:0] val,
    input logic [7:0] out_bits,
    input logic act_relu,
    input logic saturate
  );
    logic signed [ACC_W-1:0] act_val;
    logic signed [ACC_W-1:0] clamped;
    logic signed [ACC_W-1:0] rounded;
    logic signed [ACC_W-1:0] min_rep, max_rep;
    
    // Activation function
    if (act_relu && val < 0) begin
      act_val = 0;
    end else begin
      act_val = val;
    end
    
    // Calculate representable range
    case (out_bits)
      2: begin min_rep = -3; max_rep = 3; end
      4: begin min_rep = -15; max_rep = 15; end
      8: begin min_rep = -255; max_rep = 255; end
      16: begin min_rep = -32767; max_rep = 32767; end
      default: begin min_rep = -3; max_rep = 3; end
    endcase
    
    // Saturation
    if (saturate) begin
      if (act_val < min_rep) clamped = min_rep;
      else if (act_val > max_rep) clamped = max_rep;
      else clamped = act_val;
    end else begin
      clamped = act_val;
    end
    
    // Round to nearest odd (ties away from zero)
    // For MNISC-Q: values are odd numbers
    if (clamped[0] == 1'b1) begin
      // Already odd
      rounded = clamped;
    end else begin
      // Even: round away from zero
      if (clamped >= 0) begin
        rounded = clamped + 1;
      end else begin
        rounded = clamped - 1;
      end
    end
    
    // Re-clamp after rounding
    if (rounded < min_rep) rounded = min_rep;
    if (rounded > max_rep) rounded = max_rep;
    
    // Encode to MNISC-Q code
    // N-bit code = sum of slices, each slice is 2-bit
    // For now, implement 2-bit encoding directly
    // 2-bit: -3->00, -1->01, +1->10, +3->11
    case (out_bits)
      2: begin
        case (rounded)
          -3: quantize_elem = 2'b00;
          -1: quantize_elem = 2'b01;
           1: quantize_elem = 2'b10;
           3: quantize_elem = 2'b11;
          default: quantize_elem = 2'b00;
        endcase
      end
      4: begin
        // 4-bit = two 2-bit slices
        // val = a0 + a1*4 where a0, a1 are decoded 2-bit values
        // a0 = rounded mod 4 (approximately)
        // a1 = rounded / 4
        logic signed [ACC_W-1:0] low_slice, high_slice;
        logic [1:0] low_code, high_code;
        
        low_slice = rounded - ((rounded / 4) * 4);
        if (low_slice < -3) low_slice = -3;
        if (low_slice > 3) low_slice = 3;
        if (low_slice == 0) low_slice = rounded >= 0 ? 1 : -1;
        if (low_slice == 2) low_slice = rounded >= 0 ? 3 : -3;
        if (low_slice == -2) low_slice = -3;
        
        high_slice = rounded / 4;
        if (high_slice < -3) high_slice = -3;
        if (high_slice > 3) high_slice = 3;
        if (high_slice == 0) high_slice = rounded >= 0 ? 1 : -1;
        if (high_slice == 2) high_slice = rounded >= 0 ? 3 : -3;
        if (high_slice == -2) high_slice = -3;
        
        case (low_slice)
          -3: low_code = 2'b00;
          -1: low_code = 2'b01;
           1: low_code = 2'b10;
           3: low_code = 2'b11;
          default: low_code = 2'b00;
        endcase
        
        case (high_slice)
          -3: high_code = 2'b00;
          -1: high_code = 2'b01;
           1: high_code = 2'b10;
           3: high_code = 2'b11;
          default: high_code = 2'b00;
        endcase
        
        quantize_elem = {high_code, low_code};
      end
      8, 16: begin
        // For 8/16 bits, similar slice-based encoding
        // Simplified: just use lower bits
        quantize_elem = rounded[15:0];
      end
      default: quantize_elem = 0;
    endcase
  endfunction
  
  // Extract input elements
  logic signed [ACC_W-1:0] in_elems [0:BUS_W/32-1];
  logic [15:0] out_codes [0:BUS_W/2-1];
  
  always_comb begin
    for (int i = 0; i < BUS_W/32; i++) begin
      in_elems[i] = act_in_data[i*32+:32];
    end
  end
  
  // Process elements
  always_comb begin
    for (int i = 0; i < BUS_W/2; i++) begin
      if (i < in_elems_per_beat) begin
        out_codes[i] = quantize_elem(in_elems[i], cfg_out_bits, 
                                     cfg_act_fn == 1, cfg_saturate_en);
      end else begin
        out_codes[i] = 0;
      end
    end
  end
  
  // Pack output
  logic [BUS_W-1:0] packed_out;
  always_comb begin
    packed_out = 0;
    case (cfg_out_bits)
      2: begin
        for (int i = 0; i < BUS_W/2; i++) begin
          packed_out[i*2+:2] = out_codes[i][1:0];
        end
      end
      4: begin
        for (int i = 0; i < BUS_W/4; i++) begin
          packed_out[i*4+:4] = out_codes[i][3:0];
        end
      end
      8: begin
        for (int i = 0; i < BUS_W/8; i++) begin
          packed_out[i*8+:8] = out_codes[i][7:0];
        end
      end
      16: begin
        for (int i = 0; i < BUS_W/16; i++) begin
          packed_out[i*16+:16] = out_codes[i][15:0];
        end
      end
    endcase
  end
  
  // State machine
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= ST_IDLE;
      elem_cnt <= 0;
    end else begin
      case (state)
        ST_IDLE: begin
          if (start) begin
            state <= ST_PROCESS;
            elem_cnt <= 0;
          end
        end
        
        ST_PROCESS: begin
          if (act_in_valid && act_in_ready) begin
            elem_cnt <= elem_cnt + in_elems_per_beat;
            if (elem_cnt + in_elems_per_beat >= total_elems) begin
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
  
  // Output assignment
  assign out_data = packed_out;
  assign out_valid = (state == ST_PROCESS) && act_in_valid;
  assign act_in_ready = (state == ST_PROCESS) && out_ready;
  assign busy = (state != ST_IDLE);
  assign done = (state == ST_DONE);

endmodule : act_quant_unit
