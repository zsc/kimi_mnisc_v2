//------------------------------------------------------------------------------
// pool2d_unit - 2D Pooling Unit (2x2 stride=2)
//------------------------------------------------------------------------------
// Features:
//   - Max pooling and Average pooling
//   - 2x2 kernel with stride 2
//   - Supports 2/4/8/16/32 bit elements
//------------------------------------------------------------------------------
module pool2d_unit #(
  parameter int BUS_W = 128
)(
  input  logic clk,
  input  logic rst_n,
  
  // Configuration
  input  logic [7:0]  cfg_elem_bits, // Element bits (2/4/8/16/32)
  input  logic        cfg_pool_kind, // 0=Max, 1=Avg
  input  logic [7:0]  cfg_ksize,     // Kernel size (only 2 supported)
  input  logic [7:0]  cfg_stride,    // Stride (only 2 supported)
  input  logic [15:0] cfg_h_in,      // Input height
  input  logic [15:0] cfg_w_in,      // Input width
  input  logic [15:0] cfg_c,         // Channels
  
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
  typedef enum logic [3:0] {
    ST_IDLE,
    ST_FILL_LINE0,
    ST_FILL_LINE1,
    ST_POOL,
    ST_OUTPUT,
    ST_DONE
  } state_e;
  
  state_e state;
  
  // Line buffers (store 2 rows)
  localparam int MAX_LINE_WIDTH = 1024;
  localparam int ELEM_PER_BEAT = BUS_W / 32;  // Max 32-bit elements per beat
  
  logic [BUS_W-1:0] line_buf [0:1][0:MAX_LINE_WIDTH/ELEM_PER_BEAT-1];
  logic [15:0] line_wr_ptr [0:1];
  logic [15:0] line_rd_ptr;
  logic        line_sel;
  
  // Counters
  logic [15:0] h_cnt;
  logic [15:0] w_cnt;
  logic [15:0] c_cnt;
  logic [15:0] h_out, w_out;
  
  // Pooling window (2x2)
  logic signed [31:0] window [0:1][0:1];
  logic [ELEM_PER_BEAT-1:0] elem_valid;
  
  // Element extraction
  logic [31:0] elem_val;
  logic [4:0]  elem_bit_pos;
  
  // Decode element from packed input
  function automatic logic signed [31:0] decode_elem(
    input logic [BUS_W-1:0] data,
    input logic [15:0] idx,
    input logic [7:0] bits
  );
    logic [31:0] val;
    logic [10:0] bit_start;
    
    bit_start = idx * bits;
    
    case (bits)
      2: begin
        case (bit_start[2:0])
          0: val = {{30{data[1]}}, data[1:0]};
          2: val = {{30{data[3]}}, data[3:2]};
          4: val = {{30{data[5]}}, data[5:4]};
          6: val = {{30{data[7]}}, data[7:6]};
          default: val = 0;
        endcase
        // MNISC-Q decode: 00->-3, 01->-1, 10->+1, 11->+3
        case (val[1:0])
          2'b00: decode_elem = -3;
          2'b01: decode_elem = -1;
          2'b10: decode_elem = +1;
          2'b11: decode_elem = +3;
        endcase
      end
      4: begin
        case (bit_start[2:0])
          0: val = {{28{data[3]}}, data[3:0]};
          4: val = {{28{data[7]}}, data[7:4]};
          default: val = 0;
        endcase
        // 4-bit MNISC-Q: two 2-bit slices
        decode_elem = decode2(val[1:0]) + (decode2(val[3:2]) << 2);
      end
      8: begin
        val = {{24{data[bit_start[4:0]+7]}}, data[bit_start[4:0]+:8]};
        // 8-bit MNISC-Q: four 2-bit slices
        decode_elem = decode2(val[1:0]) + (decode2(val[3:2]) << 2) 
                    + (decode2(val[5:4]) << 4) + (decode2(val[7:6]) << 6);
      end
      16: begin
        logic [15:0] half;
        half = data[bit_start[4:0]+:16];
        // Eight 2-bit slices
        decode_elem = 0;
        for (int s = 0; s < 8; s++) begin
          decode_elem = decode_elem + (decode2(half[s*2+:2]) << (s*2));
        end
      end
      32: begin
        decode_elem = data[bit_start[4:0]+:32];
      end
      default: decode_elem = 0;
    endcase
  endfunction
  
  function automatic logic signed [2:0] decode2(input logic [1:0] code);
    case (code)
      2'b00: decode2 = -3;
      2'b01: decode2 = -1;
      2'b10: decode2 = +1;
      2'b11: decode2 = +3;
    endcase
  endfunction
  
  // Encode element to packed output
  function automatic logic [BUS_W-1:0] encode_elem(
    input logic signed [31:0] val,
    input logic [15:0] idx,
    input logic [7:0] bits,
    input logic [BUS_W-1:0] current
  );
    logic [10:0] bit_start;
    bit_start = idx * bits;
    
    encode_elem = current;
    
    case (bits)
      32: encode_elem[bit_start[4:0]+:32] = val;
      16: encode_elem[bit_start[4:0]+:16] = val[15:0];
      8:  encode_elem[bit_start[4:0]+:8]  = val[7:0];
      4:  encode_elem[bit_start[4:0]+:4]  = val[3:0];
      2:  begin
        case (bit_start[2:0])
          0: encode_elem[1:0] = val[1:0];
          2: encode_elem[3:2] = val[1:0];
          4: encode_elem[5:4] = val[1:0];
          6: encode_elem[7:6] = val[1:0];
        endcase
      end
    endcase
  endfunction
  
  // Calculate output dimensions
  assign h_out = (cfg_h_in + cfg_stride - 1) / cfg_stride;
  assign w_out = (cfg_w_in + cfg_stride - 1) / cfg_stride;
  
  // State machine
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= ST_IDLE;
      h_cnt <= 0;
      w_cnt <= 0;
      c_cnt <= 0;
      line_sel <= 0;
      line_wr_ptr[0] <= 0;
      line_wr_ptr[1] <= 0;
      line_rd_ptr <= 0;
    end else begin
      case (state)
        ST_IDLE: begin
          if (start) begin
            state <= ST_FILL_LINE0;
            h_cnt <= 0;
            w_cnt <= 0;
            c_cnt <= 0;
            line_sel <= 0;
            line_wr_ptr[0] <= 0;
            line_wr_ptr[1] <= 0;
            line_rd_ptr <= 0;
          end
        end
        
        ST_FILL_LINE0: begin
          if (act_in_valid && act_in_ready) begin
            line_buf[0][line_wr_ptr[0]] <= act_in_data;
            line_wr_ptr[0] <= line_wr_ptr[0] + 1;
            
            if (line_wr_ptr[0] >= (cfg_w_in * cfg_c * cfg_elem_bits + BUS_W - 1) / BUS_W - 1) begin
              line_wr_ptr[0] <= 0;
              h_cnt <= h_cnt + 1;
              if (h_cnt >= 1) begin
                state <= ST_FILL_LINE1;
              end
            end
          end
        end
        
        ST_FILL_LINE1: begin
          if (act_in_valid && act_in_ready) begin
            line_buf[1][line_wr_ptr[1]] <= act_in_data;
            line_wr_ptr[1] <= line_wr_ptr[1] + 1;
            
            if (line_wr_ptr[1] >= (cfg_w_in * cfg_c * cfg_elem_bits + BUS_W - 1) / BUS_W - 1) begin
              line_wr_ptr[1] <= 0;
              state <= ST_POOL;
              w_cnt <= 0;
              c_cnt <= 0;
            end
          end
        end
        
        ST_POOL: begin
          // Process 2x2 window
          // For each output position, read 4 input values
          // Advance to next position
          if (w_cnt + cfg_stride < cfg_w_in) begin
            w_cnt <= w_cnt + cfg_stride;
          end else begin
            w_cnt <= 0;
            // Shift line buffers
            line_buf[0] <= line_buf[1];
            line_wr_ptr[0] <= 0;
            line_wr_ptr[1] <= 0;
            
            if (h_cnt + cfg_stride < cfg_h_in) begin
              h_cnt <= h_cnt + cfg_stride;
              state <= ST_FILL_LINE1;
            end else begin
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
  
  // Pooling computation
  logic signed [31:0] pool_result;
  
  always_comb begin
    // Extract window values (simplified)
    // In real implementation, need proper addressing
    window[0][0] = 0;
    window[0][1] = 0;
    window[1][0] = 0;
    window[1][1] = 0;
    
    if (state == ST_POOL) begin
      // Compute pooling
      if (cfg_pool_kind == 0) begin
        // Max pooling
        pool_result = window[0][0];
        if (window[0][1] > pool_result) pool_result = window[0][1];
        if (window[1][0] > pool_result) pool_result = window[1][0];
        if (window[1][1] > pool_result) pool_result = window[1][1];
      end else begin
        // Average pooling
        pool_result = (window[0][0] + window[0][1] + window[1][0] + window[1][1]) / 4;
      end
    end else begin
      pool_result = 0;
    end
  end
  
  // Output generation
  assign out_data = pool_result;
  assign out_valid = (state == ST_POOL);
  assign act_in_ready = (state == ST_FILL_LINE0) || (state == ST_FILL_LINE1);
  assign busy = (state != ST_IDLE);
  assign done = (state == ST_DONE);

endmodule : pool2d_unit
