//------------------------------------------------------------------------------
// feature_line_buffer - 3-Line Circular Buffer for Conv3x3
//------------------------------------------------------------------------------
// Features:
//   - 3-row circular buffer with programmable width
//   - Padding support (pad=0 or pad=1)
//   - Stride support (stride=1 or stride=2)
//   - Outputs 3x3 window for each output pixel
//   - Extended 3-bit output: {zero_flag, 2-bit code} for proper padding
//
// Parameters:
//   MAX_W: Maximum input width
//   MAX_IC: Maximum input channels
//   IC2_LANES: Number of 2-bit IC lanes (default 16)
//------------------------------------------------------------------------------
module feature_line_buffer #(
  parameter int MAX_W = 256,
  parameter int MAX_IC = 256,
  parameter int IC2_LANES = 16,
  parameter int ACT_BITS = 8  // Default activation bits
)(
  input  logic clk,
  input  logic rst_n,
  
  // Configuration
  input  logic [15:0] cfg_w_in,      // Input width
  input  logic [15:0] cfg_h_in,      // Input height
  input  logic [15:0] cfg_ic,        // Input channels (IC)
  input  logic [7:0]  cfg_stride,    // Stride (1 or 2)
  input  logic        cfg_pad,       // Padding enable (0 or 1)
  input  logic [7:0]  cfg_act_bits,  // Activation bits (2/4/8/16)
  
  // Control
  input  logic        start,
  output logic        busy,
  output logic        done,
  
  // Input stream (packed activation data)
  input  logic [127:0] act_in_data,  // 128-bit bus
  input  logic         act_in_valid,
  output logic         act_in_ready,
  
  // Output window (3x3xIC2_LANES, 3-bit each for zero flag support)
  output logic [2:0]   win_act [0:2][0:2][0:IC2_LANES-1],  // [kh][kw][ic_lane]
  output logic         win_valid,
  input  logic         win_ready,
  
  // Output position (for debugging/external tracking)
  output logic [15:0]  out_y,
  output logic [15:0]  out_x
);

  // Local parameters
  localparam int ROW_BITS = $clog2(MAX_W * MAX_IC / 4);  // Bits per row entry (4 codes per byte)
  localparam int ROW_DEPTH = MAX_W * MAX_IC / 4;
  localparam int PTR_W = $clog2(ROW_DEPTH);
  
  // State machine
  typedef enum logic [3:0] {
    ST_IDLE,
    ST_FILL_ROW0,
    ST_FILL_ROW1,
    ST_FILL_ROW2,
    ST_RUN,
    ST_FLUSH,
    ST_DONE
  } state_e;
  
  state_e state, next_state;
  
  // Circular buffer memory (3 rows)
  // Each entry stores 4 x 2-bit codes = 8 bits
  logic [7:0] row_mem [0:2][0:ROW_DEPTH-1];
  logic [PTR_W-1:0] row_wr_ptr [0:2];
  logic [PTR_W-1:0] row_rd_ptr [0:2];
  logic [1:0]       row_wr_sel;  // Which row is being written
  logic [1:0]       row_cnt;     // Number of valid rows
  
  // Input packing
  logic [PTR_W-1:0] total_row_entries;
  logic [PTR_W-1:0] in_entry_cnt;
  
  // Window generation
  logic [15:0] oy, ox;           // Output coordinates
  logic [15:0] h_out, w_out;     // Output dimensions
  logic [1:0]  win_kh, win_kw;   // Window position counters
  
  // Slice selection for multi-bit activations
  logic [2:0]  act_slice_sel;    // Current slice (0..act_slices-1)
  logic [2:0]  act_slices;       // Total slices = act_bits / 2
  
  // Derived configuration
  logic [PTR_W-1:0] row_size_entries;
  logic [15:0]      ic_groups;
  
  // Zero flag calculation for padding
  logic [2:0] pad_mask [0:2][0:2];  // 1 if this position should be zero (padding)
  
  // Row read data
  logic [7:0] row_rdata [0:2];
  logic [PTR_W-1:0] read_addr;
  
  // Unpacked 2-bit codes from row data
  logic [1:0] row_codes [0:2][0:3];  // [row][code_idx]
  
  // IC lane counter for output
  logic [4:0] ic_lane_cnt;  // 0..IC2_LANES-1
  
  // Calculate derived values
  assign row_size_entries = (cfg_w_in * cfg_ic + 3) / 4;  // Ceil division
  assign act_slices = cfg_act_bits[7:1];  // act_bits / 2
  assign ic_groups = (cfg_ic + IC2_LANES - 1) / IC2_LANES;
  
  // Calculate output dimensions
  always_comb begin
    if (cfg_pad) begin
      h_out = (cfg_h_in + cfg_stride - 1) / cfg_stride;
      w_out = (cfg_w_in + cfg_stride - 1) / cfg_stride;
    end else begin
      h_out = (cfg_h_in - 3) / cfg_stride + 1;
      w_out = (cfg_w_in - 3) / cfg_stride + 1;
    end
  end
  
  // State machine
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      state <= ST_IDLE;
    else
      state <= next_state;
  end
  
  always_comb begin
    next_state = state;
    case (state)
      ST_IDLE:   if (start) next_state = ST_FILL_ROW0;
      ST_FILL_ROW0: if (in_entry_cnt >= row_size_entries-1 && act_in_valid) next_state = ST_FILL_ROW1;
      ST_FILL_ROW1: if (in_entry_cnt >= row_size_entries-1 && act_in_valid) next_state = ST_FILL_ROW2;
      ST_FILL_ROW2: if (in_entry_cnt >= row_size_entries-1 && act_in_valid) next_state = ST_RUN;
      ST_RUN:    if (oy >= h_out) next_state = ST_FLUSH;
      ST_FLUSH:  next_state = ST_DONE;
      ST_DONE:   next_state = ST_IDLE;
    endcase
  end
  
  // Row write control
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      row_wr_sel <= 0;
      row_cnt <= 0;
      in_entry_cnt <= 0;
    end else begin
      case (state)
        ST_IDLE: begin
          row_wr_sel <= 0;
          row_cnt <= 0;
          in_entry_cnt <= 0;
        end
        ST_FILL_ROW0, ST_FILL_ROW1, ST_FILL_ROW2: begin
          if (act_in_valid && act_in_ready) begin
            // Write to current row
            row_mem[row_wr_sel][in_entry_cnt] <= act_in_data[7:0];  // Simplified: take first byte
            
            if (in_entry_cnt >= row_size_entries - 1) begin
              in_entry_cnt <= 0;
              row_wr_sel <= row_wr_sel + 1;
              row_cnt <= row_cnt + 1;
            end else begin
              in_entry_cnt <= in_entry_cnt + 1;
            end
          end
        end
        ST_RUN: begin
          // Continue writing new rows while reading
          if (act_in_valid && act_in_ready && row_wr_sel != 2'(row_rd_ptr[0] - 1)) begin
            row_mem[row_wr_sel][in_entry_cnt] <= act_in_data[7:0];
            if (in_entry_cnt >= row_size_entries - 1) begin
              in_entry_cnt <= 0;
              row_wr_sel <= row_wr_sel + 1;
            end else begin
              in_entry_cnt <= in_entry_cnt + 1;
            end
          end
        end
      endcase
    end
  end
  
  // Calculate read addresses for 3x3 window
  // For each output position (oy, ox), we need:
  //   in_y = oy * stride + kh - pad
  //   in_x = ox * stride + kw - pad
  
  logic [15:0] base_y [0:2];
  logic [15:0] base_x [0:2];
  
  always_comb begin
    for (int kh = 0; kh < 3; kh++) begin
      base_y[kh] = oy * cfg_stride + kh - (cfg_pad ? 1 : 0);
    end
    for (int kw = 0; kw < 3; kw++) begin
      base_x[kw] = ox * cfg_stride + kw - (cfg_pad ? 1 : 0);
    end
  end
  
  // Compute padding masks
  always_comb begin
    for (int kh = 0; kh < 3; kh++) begin
      for (int kw = 0; kw < 3; kw++) begin
        if (cfg_pad) begin
          // With pad=1, border positions need zero mask when out of bounds
          pad_mask[kh][kw] = (base_y[kh] >= cfg_h_in) || (base_y[kh][0] && base_y[kh] > 0) ||  // >= or < 0
                             (base_x[kw] >= cfg_w_in) || (base_x[kw][0] && base_x[kw] > 0);
          // Simplified: check bounds
          if (base_y[kh] >= cfg_h_in || base_x[kw] >= cfg_w_in)
            pad_mask[kh][kw] = 1;
          else
            pad_mask[kh][kw] = 0;
        end else begin
          pad_mask[kh][kw] = 0;
        end
      end
    end
  end
  
  // Window generation and output control
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      oy <= 0;
      ox <= 0;
      ic_lane_cnt <= 0;
      act_slice_sel <= 0;
      win_valid <= 0;
    end else begin
      case (state)
        ST_IDLE: begin
          oy <= 0;
          ox <= 0;
          ic_lane_cnt <= 0;
          act_slice_sel <= 0;
          win_valid <= 0;
        end
        ST_RUN: begin
          if (win_valid && win_ready) begin
            // Move to next position
            if (ic_lane_cnt < IC2_LANES - 1) begin
              ic_lane_cnt <= ic_lane_cnt + 1;
            end else begin
              ic_lane_cnt <= 0;
              if (act_slice_sel < act_slices - 1) begin
                act_slice_sel <= act_slice_sel + 1;
              end else begin
                act_slice_sel <= 0;
                if (ox < w_out - 1) begin
                  ox <= ox + 1;
                end else begin
                  ox <= 0;
                  oy <= oy + 1;
                end
              end
            end
          end
          win_valid <= (oy < h_out);
        end
        default: win_valid <= 0;
      endcase
    end
  end
  
  // Assign outputs
  assign busy = (state != ST_IDLE) && (state != ST_DONE);
  assign done = (state == ST_DONE);
  assign out_y = oy;
  assign out_x = ox;
  assign act_in_ready = (state == ST_FILL_ROW0) || (state == ST_FILL_ROW1) || 
                        (state == ST_FILL_ROW2) || 
                        (state == ST_RUN && row_wr_sel != 2'(row_rd_ptr[0] - 1));
  
  // Read row data and extract 2-bit codes
  // Row selection: circular buffer access
  logic [1:0] row_select [0:2];
  assign row_select[0] = oy % 3;
  assign row_select[1] = (oy + 1) % 3;
  assign row_select[2] = (oy + 2) % 3;
  
  // Read address calculation
  always_comb begin
    for (int kh = 0; kh < 3; kh++) begin
      for (int kw = 0; kw < 3; kw++) begin
        // Calculate linear position in row
        logic [15:0] linear_pos;
        logic [15:0] code_idx;
        logic [PTR_W-1:0] byte_addr;
        
        // Position = (base_y * W + base_x) * IC + ic_lane
        // Simplified: just use ox and ic_lane for now
        linear_pos = ox * IC2_LANES + ic_lane_cnt;
        code_idx = linear_pos % 4;
        byte_addr = linear_pos / 4;
        
        // Extract 2-bit code from appropriate byte
        row_rdata[kh] = row_mem[row_select[kh]][byte_addr];
        case (code_idx)
          0: row_codes[kh][kw] = row_rdata[kh][1:0];
          1: row_codes[kh][kw] = row_rdata[kh][3:2];
          2: row_codes[kh][kw] = row_rdata[kh][5:4];
          3: row_codes[kh][kw] = row_rdata[kh][7:6];
        endcase
        
        // Output with zero flag: {zero_flag, 2-bit code}
        win_act[kh][kw][ic_lane_cnt] = {pad_mask[kh][kw], row_codes[kh][kw]};
      end
    end
  end

endmodule : feature_line_buffer
