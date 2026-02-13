//------------------------------------------------------------------------------
// conv_core_lowbit - Low-bit Conv3x3 Core with Slice Combine Support
//------------------------------------------------------------------------------
// Features:
//   - IC2_LANES=16, OC2_LANES=16 parallel MAC lanes
//   - muladd2_lut based computation (no DSP)
//   - Unsigned reduction tree with offset correction
//   - Support for act_bits/wgt_bits > 2 via slice combination
//   - Configurable SHIFT1_EN for output calibration
//------------------------------------------------------------------------------
module conv_core_lowbit #(
  parameter int IC2_LANES = 16,
  parameter int OC2_LANES = 16,
  parameter int ACC_W = 32,
  parameter int SHIFT1_EN_DEFAULT = 1
)(
  input  logic clk,
  input  logic rst_n,
  
  // Configuration
  input  logic [7:0]  cfg_act_bits,
  input  logic [7:0]  cfg_wgt_bits,
  input  logic        cfg_shift1_en,  // If 1, output is sum >> 1 (default behavior)
  input  logic [15:0] cfg_ic,         // Input channels (for N_PAIRS calculation)
  
  // Control
  input  logic        start,
  output logic        busy,
  output logic        done,
  
  // Activation input (3x3 window from line buffer)
  // Format: {zero_flag, 2-bit code} for proper padding support
  input  logic [2:0]  act_in [0:2][0:2][0:IC2_LANES-1],  // [kh][kw][ic_lane]
  input  logic        act_valid,
  output logic        act_ready,
  
  // Weight input (from weight buffer)
  input  logic [1:0]  wgt_in [0:OC2_LANES-1][0:2][0:2][0:IC2_LANES-1],  // [oc_lane][kh][kw][ic_lane]
  input  logic        wgt_valid,
  output logic        wgt_ready,
  
  // Slice selection for multi-bit computation
  input  logic [2:0]  act_slice_sel,  // Current activation slice
  input  logic [2:0]  wgt_slice_sel,  // Current weight slice
  
  // Output
  output logic signed [ACC_W-1:0] acc_out [0:OC2_LANES-1],
  output logic                    out_valid,
  input  logic                    out_ready
);

  // Local parameters
  localparam int KH = 3;
  localparam int KW = 3;
  localparam int N_PAIRS = KH * KW * IC2_LANES / 2;  // 72 pairs when IC2_LANES=16
  localparam int OFFSET = 18;
  
  // Slice counts
  logic [3:0] act_slices;
  logic [3:0] wgt_slices;
  
  assign act_slices = cfg_act_bits[7:1];  // act_bits / 2
  assign wgt_slices = cfg_wgt_bits[7:1];  // wgt_bits / 2
  
  // State machine
  typedef enum logic [2:0] {
    ST_IDLE,
    ST_COMPUTE,
    ST_REDUCE,
    ST_OUTPUT
  } state_e;
  
  state_e state;
  
  // Pipeline registers
  logic [4:0] u_values [0:OC2_LANES-1][0:N_PAIRS-1];  // LUT outputs
  
  // Accumulators for slice combination
  // For slice combine: Y_full = sum_{s,g} Conv2b(a_s, w_g) << 2(s+g)
  logic signed [ACC_W-1:0] slice_acc [0:OC2_LANES-1];
  
  // Intermediate sums
  logic [8:0]  sum_u [0:OC2_LANES-1];  // Unsigned sum (0..18*72=1296, fits in 11 bits)
  logic signed [ACC_W-1:0] sum_s [0:OC2_LANES-1];  // Signed sum after offset correction
  
  // muladd2_lut instances
  // For each OC lane and each pair of IC lanes
  genvar oc, kh, kw, ic_pair;
  generate
    for (oc = 0; oc < OC2_LANES; oc++) begin : gen_oc
      for (kh = 0; kh < KH; kh++) begin : gen_kh
        for (kw = 0; kw < KW; kw++) begin : gen_kw
          for (ic_pair = 0; ic_pair < IC2_LANES/2; ic_pair++) begin : gen_ic_pair
            
            // Pair index
            localparam int PAIR_IDX = ((kh * KW + kw) * IC2_LANES/2) + ic_pair;
            
            // Extended LUT address: {zero_a0, a0, w0, zero_a1, a1, w1}
            logic [9:0] lut_addr;
            logic [4:0] lut_out;
            
            // Extract activation and weight codes
            logic [2:0] act_ext_0, act_ext_1;  // {zero_flag, code}
            logic [1:0] wgt_0, wgt_1;
            
            assign act_ext_0 = act_in[kh][kw][ic_pair * 2];
            assign act_ext_1 = act_in[kh][kw][ic_pair * 2 + 1];
            assign wgt_0 = wgt_in[oc][kh][kw][ic_pair * 2];
            assign wgt_1 = wgt_in[oc][kh][kw][ic_pair * 2 + 1];
            
            // Construct LUT address
            assign lut_addr = {act_ext_0[2], act_ext_0[1:0], wgt_0, 
                               act_ext_1[2], act_ext_1[1:0], wgt_1};
            
            // Instantiate extended LUT
            muladd2_lut_ext u_lut (
              .addr(lut_addr),
              .u_out(lut_out)
            );
            
            // Store output
            always_ff @(posedge clk) begin
              if (state == ST_COMPUTE) begin
                u_values[oc][PAIR_IDX] <= lut_out;
              end
            end
            
          end
        end
      end
    end
  endgenerate
  
  // Reduction tree for each OC lane
  // Sum all u_values and apply offset correction
  always_comb begin
    for (int oc = 0; oc < OC2_LANES; oc++) begin
      logic [12:0] temp_sum;
      temp_sum = 0;
      for (int p = 0; p < N_PAIRS; p++) begin
        temp_sum = temp_sum + u_values[oc][p];
      end
      sum_u[oc] = temp_sum[8:0];  // Truncate to fit
      
      // Convert back to signed: sum_s = sum_u - N_PAIRS * OFFSET
      sum_s[oc] = signed'(sum_u[oc]) - signed'(N_PAIRS * OFFSET);
      
      // Apply shift if enabled
      if (cfg_shift1_en) begin
        // sum_s is already (dot_product >> 1) due to LUT design
        // But we need to multiply by 2 to get actual dot product
        // Actually: u = (pair_sum + 18) >> 1, so pair_sum = 2*u - 18
        // Total dot product = sum(pair_sum) = 2*sum_u - N_PAIRS*18
        // = 2*(sum_s + N_PAIRS*9) - N_PAIRS*18 = 2*sum_s
        // So actual dot product = 2 * sum_s
        // With SHIFT1_EN=1, we output sum_s (which is dot_product >> 1)
        acc_out[oc] = sum_s[oc];
      end else begin
        acc_out[oc] = sum_s[oc] <<< 1;  // Multiply by 2
      end
    end
  end
  
  // State machine
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= ST_IDLE;
      for (int oc = 0; oc < OC2_LANES; oc++) begin
        slice_acc[oc] <= 0;
      end
    end else begin
      case (state)
        ST_IDLE: begin
          if (start && act_valid && wgt_valid) begin
            state <= ST_COMPUTE;
            for (int oc = 0; oc < OC2_LANES; oc++) begin
              slice_acc[oc] <= 0;
            end
          end
        end
        
        ST_COMPUTE: begin
          state <= ST_REDUCE;
        end
        
        ST_REDUCE: begin
          // Add to accumulator with slice weighting
          for (int oc = 0; oc < OC2_LANES; oc++) begin
            logic signed [ACC_W-1:0] partial;
            logic [3:0] shift_amt;
            
            shift_amt = 2 * (act_slice_sel + wgt_slice_sel);
            partial = sum_s[oc] <<< shift_amt;
            slice_acc[oc] <= slice_acc[oc] + partial;
          end
          
          // Check if all slices done
          if (act_slice_sel == act_slices - 1 && wgt_slice_sel == wgt_slices - 1) begin
            state <= ST_OUTPUT;
          end else begin
            state <= ST_IDLE;  // Wait for next slice
          end
        end
        
        ST_OUTPUT: begin
          if (out_ready) begin
            state <= ST_IDLE;
          end
        end
      endcase
    end
  end
  
  // Control signals
  assign busy = (state != ST_IDLE);
  assign done = (state == ST_OUTPUT);
  assign act_ready = (state == ST_IDLE) || (state == ST_OUTPUT);
  assign wgt_ready = (state == ST_IDLE) || (state == ST_OUTPUT);
  assign out_valid = (state == ST_OUTPUT);
  
  // Output assignment
  always_comb begin
    for (int oc = 0; oc < OC2_LANES; oc++) begin
      if (state == ST_OUTPUT) begin
        acc_out[oc] = slice_acc[oc];
      end else begin
        acc_out[oc] = 0;
      end
    end
  end

endmodule : conv_core_lowbit


//------------------------------------------------------------------------------
// conv_core_lowbit_simple - Simplified version for basic 2-bit operation
//------------------------------------------------------------------------------
module conv_core_lowbit_simple #(
  parameter int IC2_LANES = 16,
  parameter int OC2_LANES = 16,
  parameter int ACC_W = 32
)(
  input  logic clk,
  input  logic rst_n,
  
  input  logic start,
  output logic busy,
  output logic done,
  
  // Inputs (standard 2-bit codes, no zero flag)
  input  logic [1:0] act_in [0:2][0:2][0:IC2_LANES-1],
  input  logic [1:0] wgt_in [0:OC2_LANES-1][0:2][0:2][0:IC2_LANES-1],
  input  logic       valid_in,
  output logic       ready_in,
  
  // Output
  output logic signed [ACC_W-1:0] acc_out [0:OC2_LANES-1],
  output logic                    out_valid,
  input  logic                    out_ready
);

  localparam int KH = 3;
  localparam int KW = 3;
  localparam int N_PAIRS = KH * KW * IC2_LANES / 2;
  
  logic [4:0] u_vals [0:OC2_LANES-1][0:N_PAIRS-1];
  logic [8:0] sum_u [0:OC2_LANES-1];
  
  // Generate LUTs
  genvar oc, kh, kw, icp;
  generate
    for (oc = 0; oc < OC2_LANES; oc++) begin
      for (kh = 0; kh < KH; kh++) begin
        for (kw = 0; kw < KW; kw++) begin
          for (icp = 0; icp < IC2_LANES/2; icp++) begin
            localparam int PIDX = ((kh * KW + kw) * IC2_LANES/2) + icp;
            
            logic [7:0] lut_addr;
            assign lut_addr = {act_in[kh][kw][icp*2], wgt_in[oc][kh][kw][icp*2],
                               act_in[kh][kw][icp*2+1], wgt_in[oc][kh][kw][icp*2+1]};
            
            muladd2_lut u_lut (.addr(lut_addr), .u_out(u_vals[oc][PIDX]));
          end
        end
      end
    end
  endgenerate
  
  // Reduction
  always_comb begin
    for (int oc = 0; oc < OC2_LANES; oc++) begin
      logic [12:0] tmp;
      tmp = 0;
      for (int p = 0; p < N_PAIRS; p++) begin
        tmp = tmp + u_vals[oc][p];
      end
      sum_u[oc] = tmp[8:0];
      acc_out[oc] = signed'(sum_u[oc]) - signed'(N_PAIRS * 18);
    end
  end
  
  // Simple control
  assign busy = valid_in;
  assign done = valid_in;
  assign ready_in = out_ready;
  assign out_valid = valid_in;

endmodule
