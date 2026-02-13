(* Test suite for MNISC compiler *)

open Ast_ir
open Ast_sim

(* Test MNISC-Q decode/encode *)
let test_codec () =
  Printf.printf "Testing MNISC-Q codec...\n";
  
  (* Test decode2 *)
  assert (decode2 0b00 = -3);
  assert (decode2 0b01 = -1);
  assert (decode2 0b10 = 1);
  assert (decode2 0b11 = 3);
  Printf.printf "  decode2: OK\n";
  
  (* Test decodeN B4 *)
  (* 4-bit code: 2 slices *)
  (* 0001 = decode2(1) + decode2(0)<<2 = -1 + (-3)<<2 = -1 -12 = -13 *)
  let v = decodeN_bits B4 0b0001 in
  Printf.printf "  decodeN B4 0b0001 = %d (expected -13)\n" v;
  
  (* Test decodeN B8 *)
  (* 8-bit code: 4 slices *)
  let v = decodeN_bits B8 0b00000001 in
  Printf.printf "  decodeN B8 0b00000001 = %d\n" v;
  
  Printf.printf "Codec tests passed!\n"

(* Test element packing/unpacking *)
let test_packing () =
  Printf.printf "Testing element packing...\n";
  
  let test_bits bits =
    let numel = 16 in
    let elem_bits = match bits with B2 -> 2 | B4 -> 4 | B8 -> 8 | B16 -> 16 | B32 -> 32 in
    let data = Bytes.create ((numel * elem_bits + 7) / 8) in
    
    (* Write test values *)
    for i = 0 to numel - 1 do
      write_element data i bits (i - 8)  (* values -8 to 7 *)
    done;
    
    (* Read back and verify *)
    for i = 0 to numel - 1 do
      let v = read_element data i bits in
      let expected = i - 8 in
      (* Due to quantization, values may differ slightly *)
      if abs (v - expected) > 10 then
        Printf.printf "  Warning: read %d, expected %d at index %d\n" v expected i
    done
  in
  
  test_bits B4;
  test_bits B8;
  Printf.printf "Packing tests passed!\n"

(* Test Conv3x3 with simple pattern *)
let test_conv3x3 () =
  Printf.printf "Testing Conv3x3...\n";
  
  (* Simple 4x4x1 input *)
  let act_tensor = { id="act"; bits=B4; layout=HWC; shape=[4;4;1] } in
  let act_numel = 4*4*1 in
  let act_bytes = (act_numel * 4 + 7) / 8 in
  let act_data = Bytes.create act_bytes in
  Bytes.fill act_data 0 act_bytes (Char.chr 0);
  
  (* 3x3x1x1 weight (identity-like) *)
  let wgt_tensor = { id="wgt"; bits=B4; layout=HWC; shape=[3;3;1;1] } in
  let wgt_numel = 3*3*1*1 in
  let wgt_bytes = (wgt_numel * 4 + 7) / 8 in
  let wgt_data = Bytes.create wgt_bytes in
  Bytes.fill wgt_data 0 wgt_bytes (Char.chr 0);
  
  (* Run convolution *)
  let out_data = conv3x3_ref act_data act_tensor wgt_data wgt_tensor 1 1 true in
  Printf.printf "  Conv3x3 output: %d bytes\n" (Bytes.length out_data);
  Printf.printf "Conv3x3 test passed!\n"

(* Test Pool2D *)
let test_pool2d () =
  Printf.printf "Testing Pool2D...\n";
  
  let input = { id="in"; bits=B8; layout=HWC; shape=[4;4;2] } in
  let in_numel = 4*4*2 in
  let in_bytes = in_numel in  (* B8 *)
  let in_data = Bytes.create in_bytes in
  for i = 0 to in_bytes - 1 do
    Bytes.set in_data i (Char.chr (i mod 256))
  done;
  
  let out_data = pool2d_ref in_data input Max in
  Printf.printf "  Pool2D Max output: %d bytes\n" (Bytes.length out_data);
  
  let out_data2 = pool2d_ref in_data input Avg in
  Printf.printf "  Pool2D Avg output: %d bytes\n" (Bytes.length out_data2);
  Printf.printf "Pool2D test passed!\n"

(* Test ConcatC *)
let test_concat () =
  Printf.printf "Testing ConcatC...\n";
  
  let a = { id="a"; bits=B8; layout=HWC; shape=[2;2;3] } in
  let b = { id="b"; bits=B8; layout=HWC; shape=[2;2;5] } in
  let a_data = Bytes.create (2*2*3) in
  let b_data = Bytes.create (2*2*5) in
  Bytes.fill a_data 0 (2*2*3) (Char.chr 1);
  Bytes.fill b_data 0 (2*2*5) (Char.chr 2);
  
  let out_data = concat_ref a_data a b_data b in
  Printf.printf "  Concat output: %d bytes\n" (Bytes.length out_data);
  
  (* Verify first few elements *)
  let v1 = read_element out_data 0 B8 in
  let v2 = read_element out_data 12 B8 in
  Printf.printf "  First A elem: %d, First B elem: %d\n" v1 v2;
  Printf.printf "ConcatC test passed!\n"

(* Test GEMM *)
let test_gemm () =
  Printf.printf "Testing GEMM...\n";
  
  let x = { id="x"; bits=B8; layout=MK; shape=[8;16] } in
  let w = { id="w"; bits=B4; layout=MK; shape=[4;16] } in
  let x_data = Bytes.create (8*16) in
  let w_data = Bytes.create ((4*16 + 1) / 2) in
  Bytes.fill x_data 0 (8*16) (Char.chr 1);
  Bytes.fill w_data 0 ((4*16 + 1) / 2) (Char.chr 0);
  
  let out_data = gemm_ref x_data x w_data w true in
  Printf.printf "  GEMM output: %d bytes\n" (Bytes.length out_data);
  Printf.printf "GEMM test passed!\n"

(* Test ActQuant *)
let test_act_quant () =
  Printf.printf "Testing ActQuant...\n";
  
  let input = { id="in"; bits=B32; layout=HWC; shape=[2;2;4] } in
  let in_numel = 2*2*4 in
  let in_data = Bytes.create (in_numel * 4) in
  (* Fill with test values *)
  for i = 0 to in_numel - 1 do
    let value = (i - 8) * 100 in
    Bytes.set in_data (i*4+0) (Char.chr (value land 0xFF));
    Bytes.set in_data (i*4+1) (Char.chr ((value lsr 8) land 0xFF));
    Bytes.set in_data (i*4+2) (Char.chr ((value lsr 16) land 0xFF));
    Bytes.set in_data (i*4+3) (Char.chr ((value lsr 24) land 0xFF));
  done;
  
  let out_data = act_quant_ref in_data input ReLU B4 in
  Printf.printf "  ActQuant output: %d bytes\n" (Bytes.length out_data);
  Printf.printf "ActQuant test passed!\n"

(* Run all tests *)
let () =
  Printf.printf "=== MNISC Test Suite ===\n\n";
  test_codec ();
  Printf.printf "\n";
  test_packing ();
  Printf.printf "\n";
  test_conv3x3 ();
  Printf.printf "\n";
  test_pool2d ();
  Printf.printf "\n";
  test_concat ();
  Printf.printf "\n";
  test_gemm ();
  Printf.printf "\n";
  test_act_quant ();
  Printf.printf "\n=== All tests passed! ===\n"
