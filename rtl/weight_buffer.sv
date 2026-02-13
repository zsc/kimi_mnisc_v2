//------------------------------------------------------------------------------
// weight_buffer - Weight Storage Buffer for Conv3x3 and GEMM
//------------------------------------------------------------------------------
// Stores weights for one tile/operation
// Layout: [kh][kw][OC][IC] packed
//
// For Conv3x3: KH=3, KW=3
// For GEMM: treat as KH=1, KW=1, IC=K, OC=N
//------------------------------------------------------------------------------
module weight_buffer #(
  parameter int MAX_OC = 256,
  parameter int MAX_IC = 256,
  parameter int IC2_LANES = 16,
  parameter int OC2_LANES = 16,
  parameter int WBUF_BYTES = 8192  // 8KB weight buffer
)(
  input  logic clk,
  input  logic rst_n,
  
  // Configuration
  input  logic [15:0] cfg_oc,        // Output channels
  input  logic [15:0] cfg_ic,        // Input channels
  input  logic [7:0]  cfg_kh,        // Kernel height (3 for Conv3x3, 1 for GEMM)
  input  logic [7:0]  cfg_kw,        // Kernel width (3 for Conv3x3, 1 for GEMM)
  input  logic [7:0]  cfg_wgt_bits,  // Weight bits (2/4/8/16)
  
  // Control
  input  logic        load_start,
  output logic        load_busy,
  output logic        load_done,
  input  logic [31:0] cfg_wgt_bytes, // Total weight bytes to load
  
  // Weight input stream
  input  logic [127:0] wgt_in_data,
  input  logic         wgt_in_valid,
  output logic         wgt_in_ready,
  
  // Weight output (organized for MAC array)
  // For each OC2_LANE, we output weights for all IC2_LANES
  output logic [1:0]   wgt_out [0:OC2_LANES-1][0:2][0:2][0:IC2_LANES-1],  // [oc_lane][kh][kw][ic_lane]
  output logic         wgt_out_valid,
  input  logic         wgt_out_ready,
  
  // Slice selection for multi-bit weights
  input  logic [2:0]   wgt_slice_sel  // Which 2-bit slice (0..wgt_slices-1)
);

  // Local parameters
  localparam int WBUF_DEPTH = WBUF_BYTES;
  localparam int ADDR_W = $clog2(WBUF_DEPTH);
  
  // Weight buffer memory (byte addressable)
  logic [7:0] wbuf [0:WBUF_DEPTH-1];
  
  // State
  typedef enum logic [2:0] {
    ST_IDLE,
    ST_LOADING,
    ST_LOAD_DONE,
    ST_READ
  } state_e;
  
  state_e state;
  
  // Loading counters
  logic [ADDR_W-1:0] load_addr;
  logic [31:0]       load_cnt;
  
  // Reading counters
  logic [15:0] oc_tile_cnt;   // Which OC tile (0, 16, 32, ...)
  logic [15:0] ic_tile_cnt;   // Which IC tile
  logic [7:0]  kh_cnt, kw_cnt;
  logic [2:0]  wgt_slices;
  
  // Calculate derived values
  assign wgt_slices = cfg_wgt_bits[7:1];  // wgt_bits / 2
  
  // Address calculation for reading
  // Linear layout: (((kh * KW + kw) * OC + oc) * IC + ic) * wgt_bits / 8
  function automatic logic [ADDR_W-1:0] calc_wgt_addr(
    input logic [7:0]  kh,
    input logic [7:0]  kw,
    input logic [15:0] oc,
    input logic [15:0] ic,
    input logic [2:0]  slice_sel
  );
    logic [31:0] linear_elem;
    logic [31:0] bit_offset;
    logic [ADDR_W-1:0] byte_addr;
    
    linear_elem = ((kh * cfg_kw + kw) * cfg_oc + oc) * cfg_ic + ic;
    bit_offset = linear_elem * cfg_wgt_bits + slice_sel * 2;
    byte_addr = bit_offset[ADDR_W+2:3];  // Divide by 8
    
    return byte_addr;
  endfunction
  
  // State machine
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= ST_IDLE;
      load_addr <= 0;
      load_cnt <= 0;
      oc_tile_cnt <= 0;
      ic_tile_cnt <= 0;
      kh_cnt <= 0;
      kw_cnt <= 0;
    end else begin
      case (state)
        ST_IDLE: begin
          if (load_start) begin
            state <= ST_LOADING;
            load_addr <= 0;
            load_cnt <= 0;
          end
        end
        
        ST_LOADING: begin
          if (wgt_in_valid && wgt_in_ready) begin
            wbuf[load_addr] <= wgt_in_data[7:0];  // Simplified: store first byte
            load_addr <= load_addr + 1;
            load_cnt <= load_cnt + 1;
            
            if (load_cnt >= cfg_wgt_bytes - 1) begin
              state <= ST_LOAD_DONE;
            end
          end
        end
        
        ST_LOAD_DONE: begin
          state <= ST_READ;
          oc_tile_cnt <= 0;
          ic_tile_cnt <= 0;
          kh_cnt <= 0;
          kw_cnt <= 0;
        end
        
        ST_READ: begin
          if (wgt_out_valid && wgt_out_ready) begin
            // Advance counters
            if (kw_cnt < cfg_kw - 1) begin
              kw_cnt <= kw_cnt + 1;
            end else begin
              kw_cnt <= 0;
              if (kh_cnt < cfg_kh - 1) begin
                kh_cnt <= kh_cnt + 1;
              end else begin
                kh_cnt <= 0;
                if (oc_tile_cnt + OC2_LANES < cfg_oc) begin
                  oc_tile_cnt <= oc_tile_cnt + OC2_LANES;
                end else begin
                  oc_tile_cnt <= 0;
                  // Done with this read batch
                  state <= ST_IDLE;
                end
              end
            end
          end
        end
      endcase
    end
  end
  
  // Output control
  assign load_busy = (state == ST_LOADING);
  assign load_done = (state == ST_LOAD_DONE);
  assign wgt_in_ready = (state == ST_LOADING);
  assign wgt_out_valid = (state == ST_READ);
  
  // Extract weights from buffer
  // For each OC2_LANE, read corresponding weight
  always_comb begin
    for (int oc_lane = 0; oc_lane < OC2_LANES; oc_lane++) begin
      for (int kh = 0; kh < 3; kh++) begin
        for (int kw = 0; kw < 3; kw++) begin
          for (int ic_lane = 0; ic_lane < IC2_LANES; ic_lane++) begin
            logic [ADDR_W-1:0] addr;
            logic [7:0] rdata;
            logic [3:0] bit_pos;
            
            addr = calc_wgt_addr(kh[7:0], kw[7:0], 
                                 oc_tile_cnt + oc_lane[15:0], 
                                 ic_tile_cnt + ic_lane[15:0],
                                 wgt_slice_sel);
            rdata = wbuf[addr];
            
            // Extract 2-bit code from appropriate position
            bit_pos = (calc_bit_offset(oc_lane, ic_lane) % 8) / 2;
            case (bit_pos)
              0: wgt_out[oc_lane][kh][kw][ic_lane] = rdata[1:0];
              1: wgt_out[oc_lane][kh][kw][ic_lane] = rdata[3:2];
              2: wgt_out[oc_lane][kh][kw][ic_lane] = rdata[5:4];
              3: wgt_out[oc_lane][kh][kw][ic_lane] = rdata[7:6];
              default: wgt_out[oc_lane][kh][kw][ic_lane] = 2'b00;
            endcase
          end
        end
      end
    end
  end
  
  // Helper function for bit offset calculation
  function automatic logic [7:0] calc_bit_offset(
    input int oc_lane,
    input int ic_lane
  );
    logic [31:0] linear_elem;
    linear_elem = ((kh_cnt * cfg_kw + kw_cnt) * cfg_oc + oc_tile_cnt + oc_lane) 
                  * cfg_ic + ic_tile_cnt + ic_lane;
    return linear_elem[7:0] * cfg_wgt_bits[7:0];
  endfunction

endmodule : weight_buffer
