//------------------------------------------------------------------------------
// gemm_core_lowbit - Low-bit GEMM (Matrix Multiplication) Core
//------------------------------------------------------------------------------
// Computes Y = X * W^T where:
//   X: [M, K] matrix (activations)
//   W: [N, K] matrix (weights, transposed for computation)
//   Y: [M, N] matrix (output)
//
// Features:
//   - K-dimension dot product using muladd2_lut
//   - Same slice combine logic as conv_core for multi-bit support
//   - Supports IC2_LANES parallel K computation
//------------------------------------------------------------------------------
module gemm_core_lowbit #(
  parameter int IC2_LANES = 16,  // K parallelism (K per cycle)
  parameter int OC2_LANES = 16,  // N parallelism
  parameter int ACC_W = 32,
  parameter int SHIFT1_EN_DEFAULT = 1
)(
  input  logic clk,
  input  logic rst_n,
  
  // Configuration
  input  logic [15:0] cfg_m,         // Number of rows in X / rows in Y
  input  logic [15:0] cfg_k,         // Number of columns in X / columns in W
  input  logic [15:0] cfg_n,         // Number of columns in Y / rows in W
  input  logic [7:0]  cfg_act_bits,  // X bits (2/4/8/16)
  input  logic [7:0]  cfg_wgt_bits,  // W bits (2/4/8/16)
  input  logic        cfg_shift1_en,
  
  // Control
  input  logic        start,
  output logic        busy,
  output logic        done,
  
  // Activation input (K elements per cycle)
  // Format: {zero_flag, 2-bit code} for padding support
  input  logic [2:0]  act_in [0:IC2_LANES-1],  // X[m, k:k+15]
  input  logic        act_valid,
  output logic        act_ready,
  input  logic [15:0] act_k_offset,  // Current K position
  
  // Weight input (N x K elements)
  // W[n, k:k+15] for each n in parallel
  input  logic [1:0]  wgt_in [0:OC2_LANES-1][0:IC2_LANES-1],  // [n_lane][k_lane]
  input  logic        wgt_valid,
  output logic        wgt_ready,
  
  // Slice selection
  input  logic [2:0]  act_slice_sel,
  input  logic [2:0]  wgt_slice_sel,
  
  // Output
  output logic signed [ACC_W-1:0] acc_out [0:OC2_LANES-1],
  output logic                    out_valid,
  input  logic                    out_ready,
  output logic [15:0]             out_m_idx,
  output logic [15:0]             out_n_idx
);

  // Local parameters
  localparam int N_PAIRS = IC2_LANES / 2;  // 8 pairs when IC2_LANES=16
  localparam int OFFSET = 18;
  
  // Slice counts
  logic [3:0] act_slices;
  logic [3:0] wgt_slices;
  
  assign act_slices = cfg_act_bits[7:1];
  assign wgt_slices = cfg_wgt_bits[7:1];
  
  // State machine
  typedef enum logic [3:0] {
    ST_IDLE,
    ST_LOAD_K,      // Loading K dimension
    ST_COMPUTE,     // Compute dot product for current K tile
    ST_REDUCE,      // Add to accumulator
    ST_OUTPUT,      // Output result
    ST_NEXT_MN      // Move to next M/N tile
  } state_e;
  
  state_e state;
  
  // Counters
  logic [15:0] m_cnt;
  logic [15:0] n_cnt;
  logic [15:0] k_cnt;
  logic [15:0] k_tiles;
  
  // Accumulators
  logic signed [ACC_W-1:0] accumulators [0:OC2_LANES-1];
  
  // Pipeline registers
  logic [4:0] u_values [0:OC2_LANES-1][0:N_PAIRS-1];
  logic [8:0] sum_u [0:OC2_LANES-1];
  logic signed [ACC_W-1:0] sum_s [0:OC2_LANES-1];
  
  // Calculate K tiles
  assign k_tiles = (cfg_k + IC2_LANES - 1) / IC2_LANES;
  
  // muladd2_lut instances for GEMM
  // For each OC lane and each pair of IC lanes
  genvar oc, ic_pair;
  generate
    for (oc = 0; oc < OC2_LANES; oc++) begin : gen_oc
      for (ic_pair = 0; ic_pair < IC2_LANES/2; ic_pair++) begin : gen_ic_pair
        
        logic [9:0] lut_addr;
        logic [4:0] lut_out;
        
        // Extract activation and weight codes
        logic [2:0] act_ext_0, act_ext_1;
        logic [1:0] wgt_0, wgt_1;
        
        assign act_ext_0 = act_in[ic_pair * 2];
        assign act_ext_1 = act_in[ic_pair * 2 + 1];
        assign wgt_0 = wgt_in[oc][ic_pair * 2];
        assign wgt_1 = wgt_in[oc][ic_pair * 2 + 1];
        
        // Construct LUT address for extended LUT
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
            u_values[oc][ic_pair] <= lut_out;
          end
        end
        
      end
    end
  endgenerate
  
  // Reduction tree for each OC lane
  always_comb begin
    for (int oc = 0; oc < OC2_LANES; oc++) begin
      logic [12:0] temp_sum;
      temp_sum = 0;
      for (int p = 0; p < N_PAIRS; p++) begin
        temp_sum = temp_sum + u_values[oc][p];
      end
      sum_u[oc] = temp_sum[8:0];
      
      // Convert back to signed
      sum_s[oc] = signed'(sum_u[oc]) - signed'(N_PAIRS * OFFSET);
    end
  end
  
  // State machine
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= ST_IDLE;
      m_cnt <= 0;
      n_cnt <= 0;
      k_cnt <= 0;
      for (int oc = 0; oc < OC2_LANES; oc++) begin
        accumulators[oc] <= 0;
      end
    end else begin
      case (state)
        ST_IDLE: begin
          if (start) begin
            state <= ST_LOAD_K;
            m_cnt <= 0;
            n_cnt <= 0;
            k_cnt <= 0;
            for (int oc = 0; oc < OC2_LANES; oc++) begin
              accumulators[oc] <= 0;
            end
          end
        end
        
        ST_LOAD_K: begin
          if (act_valid && wgt_valid) begin
            state <= ST_COMPUTE;
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
            
            // Apply shift1 if enabled
            if (cfg_shift1_en) begin
              partial = sum_s[oc];
            end else begin
              partial = sum_s[oc] <<< 1;
            end
            
            // Apply slice shift
            partial = partial <<< shift_amt;
            
            accumulators[oc] <= accumulators[oc] + partial;
          end
          
          // Check if all slices and K tiles done
          if (act_slice_sel == act_slices - 1 && wgt_slice_sel == wgt_slices - 1) begin
            if (k_cnt + IC2_LANES >= cfg_k) begin
              state <= ST_OUTPUT;
            end else begin
              k_cnt <= k_cnt + IC2_LANES;
              state <= ST_LOAD_K;
            end
          end else begin
            state <= ST_LOAD_K;  // Next slice
          end
        end
        
        ST_OUTPUT: begin
          if (out_ready) begin
            state <= ST_NEXT_MN;
          end
        end
        
        ST_NEXT_MN: begin
          // Clear accumulators for next tile
          for (int oc = 0; oc < OC2_LANES; oc++) begin
            accumulators[oc] <= 0;
          end
          k_cnt <= 0;
          
          // Advance to next tile
          if (n_cnt + OC2_LANES < cfg_n) begin
            n_cnt <= n_cnt + OC2_LANES;
            state <= ST_LOAD_K;
          end else if (m_cnt + 1 < cfg_m) begin
            m_cnt <= m_cnt + 1;
            n_cnt <= 0;
            state <= ST_LOAD_K;
          end else begin
            state <= ST_IDLE;  // All done
          end
        end
      endcase
    end
  end
  
  // Output assignment
  always_comb begin
    for (int oc = 0; oc < OC2_LANES; oc++) begin
      acc_out[oc] = accumulators[oc];
    end
  end
  
  // Control signals
  assign busy = (state != ST_IDLE);
  assign done = (state == ST_IDLE) && !start;
  assign act_ready = (state == ST_LOAD_K);
  assign wgt_ready = (state == ST_LOAD_K);
  assign out_valid = (state == ST_OUTPUT);
  assign out_m_idx = m_cnt;
  assign out_n_idx = n_cnt;

endmodule : gemm_core_lowbit
