//------------------------------------------------------------------------------
// unpool2d_unit - 2D Upsampling Unit (2x Nearest Neighbor)
//------------------------------------------------------------------------------
// Features:
//   - Nearest neighbor upsampling (repeat each element 2x2)
//   - Scale factor 2 (output H=2*H_in, W=2*W_in)
//   - Supports 2/4/8/16/32 bit elements
//------------------------------------------------------------------------------
module unpool2d_unit #(
  parameter int BUS_W = 128
)(
  input  logic clk,
  input  logic rst_n,
  
  // Configuration
  input  logic [7:0]  cfg_elem_bits,  // Element bits (2/4/8/16/32)
  input  logic [7:0]  cfg_unpool_kind, // 0=nearest_repeat
  input  logic [7:0]  cfg_scale,       // Scale factor (only 2 supported)
  input  logic [15:0] cfg_h_in,        // Input height
  input  logic [15:0] cfg_w_in,        // Input width
  input  logic [15:0] cfg_c,           // Channels
  
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
    ST_LOAD_LINE,
    ST_OUTPUT_ROW0,
    ST_OUTPUT_ROW1,
    ST_NEXT_LINE,
    ST_DONE
  } state_e;
  
  state_e state;
  
  // Line buffer (store one input row)
  localparam int MAX_LINE_WIDTH = 2048;  // In bytes
  localparam int LINE_BUF_DEPTH = MAX_LINE_WIDTH * 4 / BUS_W;  // For 32-bit elements
  
  logic [BUS_W-1:0] line_buf [0:LINE_BUF_DEPTH-1];
  logic [15:0] line_wr_ptr;
  logic [15:0] line_rd_ptr;
  
  // Counters
  logic [15:0] h_in_cnt;   // Current input row
  logic [15:0] h_out_cnt;  // Current output row (0 or 1 within each input row)
  logic [15:0] w_out_cnt;  // Current output column
  logic [15:0] c_cnt;      // Current channel
  
  // Output row buffer
  logic [BUS_W-1:0] out_row_buf;
  logic [15:0]      out_bit_cnt;  // Bit position in output
  
  // Calculate derived values
  logic [15:0] h_out, w_out;
  logic [15:0] elems_per_row;
  logic [15:0] beats_per_row;
  
  assign h_out = cfg_h_in * cfg_scale;
  assign w_out = cfg_w_in * cfg_scale;
  assign elems_per_row = cfg_w_in * cfg_c;
  assign beats_per_row = (elems_per_row * cfg_elem_bits + BUS_W - 1) / BUS_W;
  
  // Element extraction and duplication
  logic [31:0] current_elem;
  logic [15:0] elem_idx;
  logic [4:0]  bit_offset;
  
  // Extract element from input
  function automatic logic [31:0] extract_elem(
    input logic [BUS_W-1:0] data,
    input logic [15:0] elem_idx,
    input logic [7:0] bits
  );
    logic [15:0] bit_start;
    bit_start = elem_idx * bits;
    
    case (bits)
      2: begin
        case (bit_start[2:0])
          0: extract_elem = {{30{1'b0}}, data[1:0]};
          2: extract_elem = {{30{1'b0}}, data[3:2]};
          4: extract_elem = {{30{1'b0}}, data[5:4]};
          6: extract_elem = {{30{1'b0}}, data[7:6]};
          default: extract_elem = 0;
        endcase
      end
      4: begin
        case (bit_start[2:0])
          0: extract_elem = {{28{1'b0}}, data[3:0]};
          4: extract_elem = {{28{1'b0}}, data[7:4]};
          default: extract_elem = 0;
        endcase
      end
      8:  extract_elem = {{24{1'b0}}, data[bit_start[6:0]+:8]};
      16: extract_elem = {{16{1'b0}}, data[bit_start[6:0]+:16]};
      32: extract_elem = data[bit_start[6:0]+:32];
      default: extract_elem = 0;
    endcase
  endfunction
  
  // Insert element into output
  function automatic logic [BUS_W-1:0] insert_elem(
    input logic [BUS_W-1:0] current,
    input logic [31:0] elem,
    input logic [15:0] elem_idx,
    input logic [7:0] bits
  );
    logic [15:0] bit_start;
    bit_start = elem_idx * bits;
    
    insert_elem = current;
    case (bits)
      2: begin
        case (bit_start[2:0])
          0: insert_elem[1:0] = elem[1:0];
          2: insert_elem[3:2] = elem[1:0];
          4: insert_elem[5:4] = elem[1:0];
          6: insert_elem[7:6] = elem[1:0];
        endcase
      end
      4: begin
        case (bit_start[2:0])
          0: insert_elem[3:0] = elem[3:0];
          4: insert_elem[7:4] = elem[3:0];
        endcase
      end
      8:  insert_elem[bit_start[6:0]+:8]  = elem[7:0];
      16: insert_elem[bit_start[6:0]+:16] = elem[15:0];
      32: insert_elem[bit_start[6:0]+:32] = elem;
    endcase
  endfunction
  
  // State machine
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= ST_IDLE;
      h_in_cnt <= 0;
      h_out_cnt <= 0;
      w_out_cnt <= 0;
      line_wr_ptr <= 0;
      line_rd_ptr <= 0;
      out_bit_cnt <= 0;
      out_row_buf <= 0;
    end else begin
      case (state)
        ST_IDLE: begin
          if (start) begin
            state <= ST_LOAD_LINE;
            h_in_cnt <= 0;
            h_out_cnt <= 0;
            w_out_cnt <= 0;
            line_wr_ptr <= 0;
            line_rd_ptr <= 0;
            out_bit_cnt <= 0;
            out_row_buf <= 0;
          end
        end
        
        ST_LOAD_LINE: begin
          if (act_in_valid && act_in_ready) begin
            line_buf[line_wr_ptr] <= act_in_data;
            line_wr_ptr <= line_wr_ptr + 1;
            
            if (line_wr_ptr >= beats_per_row - 1) begin
              line_wr_ptr <= 0;
              line_rd_ptr <= 0;
              state <= ST_OUTPUT_ROW0;
              w_out_cnt <= 0;
              out_bit_cnt <= 0;
              out_row_buf <= 0;
            end
          end
        end
        
        ST_OUTPUT_ROW0, ST_OUTPUT_ROW1: begin
          // Generate output by repeating each input element twice
          if (out_ready || !out_valid) begin
            // Calculate which input element to read
            logic [15:0] in_elem_idx;
            logic [15:0] out_elem_idx;
            
            in_elem_idx = w_out_cnt / cfg_scale;
            out_elem_idx = w_out_cnt;
            
            // Extract element from line buffer
            // Simplified: assume elements fit in 32 bits
            current_elem = extract_elem(line_buf[in_elem_idx / (BUS_W/cfg_elem_bits)],
                                        in_elem_idx % (BUS_W/cfg_elem_bits),
                                        cfg_elem_bits);
            
            // Duplicate element in output (nearest neighbor)
            // For scale=2: each input element becomes 2x2 output elements
            // We're generating rows, so each row has 2x elements
            out_row_buf <= insert_elem(out_row_buf, current_elem, out_elem_idx, cfg_elem_bits);
            
            w_out_cnt <= w_out_cnt + 1;
            out_bit_cnt <= out_bit_cnt + cfg_elem_bits;
            
            if (out_bit_cnt + cfg_elem_bits >= BUS_W) begin
              // Output beat is ready
              out_bit_cnt <= 0;
            end
            
            if (w_out_cnt >= elems_per_row * cfg_scale - 1) begin
              w_out_cnt <= 0;
              if (state == ST_OUTPUT_ROW0) begin
                state <= ST_OUTPUT_ROW1;
                line_rd_ptr <= 0;
              end else begin
                state <= ST_NEXT_LINE;
              end
            end
          end
        end
        
        ST_NEXT_LINE: begin
          h_in_cnt <= h_in_cnt + 1;
          if (h_in_cnt + 1 < cfg_h_in) begin
            state <= ST_LOAD_LINE;
            line_wr_ptr <= 0;
          end else begin
            state <= ST_DONE;
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
  assign out_data = out_row_buf;
  assign out_valid = ((state == ST_OUTPUT_ROW0) || (state == ST_OUTPUT_ROW1)) 
                     && (out_bit_cnt + cfg_elem_bits >= BUS_W);
  assign act_in_ready = (state == ST_LOAD_LINE);
  assign busy = (state != ST_IDLE);
  assign done = (state == ST_DONE);

endmodule : unpool2d_unit
