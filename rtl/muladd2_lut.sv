//------------------------------------------------------------------------------
// muladd2_lut - 2-bit Multiplier-Adder Lookup Table
//------------------------------------------------------------------------------
// Input:  8 bits = {a0[1:0], w0[1:0], a1[1:0], w1[1:0]}
// Output: 5 bits unsigned
//
// Operation:
//   p0 = decode2(a0) * decode2(w0)
//   p1 = decode2(a1) * decode2(w1)
//   pair_sum = p0 + p1  // range [-18, +18], always even
//   OFFSET = 18
//   u = (pair_sum + OFFSET) >> 1  // range [0, 18], fits in 5 bits
//
// Note: Must use case statement, no DSP blocks allowed
//------------------------------------------------------------------------------
module muladd2_lut (
  input  logic [7:0] addr,   // {a0, w0, a1, w1}
  output logic [4:0] u_out   // unsigned output, range 0..18
);

  // Internal signals for decode
  logic signed [2:0] a0_val, w0_val, a1_val, w1_val;
  logic signed [4:0] p0, p1;        // product range: -9 to +9
  logic signed [5:0] pair_sum;      // sum range: -18 to +18
  
  // Decode 2-bit codes to signed values
  // 00->-3, 01->-1, 10->+1, 11->+3
  function automatic logic signed [2:0] decode2_fn(input logic [1:0] code);
    case (code)
      2'b00: decode2_fn = -3;
      2'b01: decode2_fn = -1;
      2'b10: decode2_fn = +1;
      2'b11: decode2_fn = +3;
      default: decode2_fn = 0;
    endcase
  endfunction

  // Extract fields from address
  logic [1:0] a0_code, w0_code, a1_code, w1_code;
  assign a0_code = addr[7:6];
  assign w0_code = addr[5:4];
  assign a1_code = addr[3:2];
  assign w1_code = addr[1:0];

  // Decode to signed values
  assign a0_val = decode2_fn(a0_code);
  assign w0_val = decode2_fn(w0_code);
  assign a1_val = decode2_fn(a1_code);
  assign w1_val = decode2_fn(w1_code);

  // Compute products
  assign p0 = a0_val * w0_val;  // range [-9, +9]
  assign p1 = a1_val * w1_val;  // range [-9, +9]

  // Sum and convert to unsigned
  assign pair_sum = p0 + p1;                    // range [-18, +18]
  assign u_out = (pair_sum + 6'sd18) >>> 1;     // range [0, 18]

endmodule : muladd2_lut


//------------------------------------------------------------------------------
// muladd2_lut_ext - Extended LUT with zero flag support
//------------------------------------------------------------------------------
// Input:  12 bits = {zero0, a0[1:0], zero_w0, w0[1:0], zero1, a1[1:0], zero_w1, w1[1:0]}
// But we optimize to: {zero_a0, a0, zero_w0, w0, zero_a1, a1, zero_w1, w1}
// Actually for feature line buffer: we only need zero flag on activation
// So simplified: {zero0, a0, w0, zero1, a1, w1} = 10 bits
//
// Let's use: {zero_a0, a0[1:0], w0[1:0], zero_a1, a1[1:0], w1[1:0]} = 10 bits
//------------------------------------------------------------------------------
module muladd2_lut_ext (
  input  logic [9:0] addr,   // {zero_a0, a0, w0, zero_a1, a1, w1}
  output logic [4:0] u_out   // unsigned output, range 0..18
);

  logic zero_a0, zero_a1;
  logic [1:0] a0_code, w0_code, a1_code, w1_code;
  logic signed [2:0] a0_val, w0_val, a1_val, w1_val;
  logic signed [4:0] p0, p1;
  logic signed [5:0] pair_sum;

  // Extract fields
  assign zero_a0 = addr[9];
  assign a0_code = addr[8:7];
  assign w0_code = addr[6:5];
  assign zero_a1 = addr[4];
  assign a1_code = addr[3:2];
  assign w1_code = addr[1:0];

  // Decode with zero flag support
  function automatic logic signed [2:0] decode2_fn(input logic [1:0] code);
    case (code)
      2'b00: decode2_fn = -3;
      2'b01: decode2_fn = -1;
      2'b10: decode2_fn = +1;
      2'b11: decode2_fn = +3;
      default: decode2_fn = 0;
    endcase
  endfunction

  // Activation with zero flag
  assign a0_val = zero_a0 ? 3'sd0 : decode2_fn(a0_code);
  assign a1_val = zero_a1 ? 3'sd0 : decode2_fn(a1_code);
  
  // Weights (no zero flag needed)
  assign w0_val = decode2_fn(w0_code);
  assign w1_val = decode2_fn(w1_code);

  // Compute products
  assign p0 = a0_val * w0_val;
  assign p1 = a1_val * w1_val;

  // Sum and convert to unsigned
  assign pair_sum = p0 + p1;
  assign u_out = (pair_sum + 6'sd18) >>> 1;

endmodule : muladd2_lut_ext
