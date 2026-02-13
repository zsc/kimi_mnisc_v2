(* AST to EU ISA lowering *)

open Ast_ir
open Eu_isa

let flags_default = 0x02  (* SHIFT1_EN = 1 *)
let flags_check_counts = 0x03  (* CHECK_COUNTS_EN | SHIFT1_EN *)

let bits_to_u8 (b:bits) : int =
  match b with B2 -> 2 | B4 -> 4 | B8 -> 8 | B16 -> 16 | B32 -> 32

let make_mode_bits ~(act_bits:int) ~(wgt_bits:int) ~(stride:int) ~(pad:int) : int32 =
  let a = Int32.of_int (act_bits land 0xFF) in
  let w = Int32.of_int ((wgt_bits land 0xFF) lsl 8) in
  let s = Int32.of_int ((stride land 0xFF) lsl 16) in
  let p = Int32.of_int ((pad land 0xFF) lsl 24) in
  Int32.(add (add a w) (add s p))

let make_shape0 ~(h_in:int) ~(w_in:int) : int32 =
  Int32.(add (of_int (h_in land 0xFFFF)) (shift_left (of_int (w_in land 0xFFFF)) 16))

let make_shape1 ~(ic:int) ~(oc:int) : int32 =
  Int32.(add (of_int (ic land 0xFFFF)) (shift_left (of_int (oc land 0xFFFF)) 16))

let make_tile0 ~(y0:int) ~(x0:int) : int32 =
  Int32.(add (of_int (y0 land 0xFFFF)) (shift_left (of_int (x0 land 0xFFFF)) 16))

let make_tile1 ~(oh:int) ~(ow:int) : int32 =
  Int32.(add (of_int (oh land 0xFFFF)) (shift_left (of_int (ow land 0xFFFF)) 16))

let make_meta ~(in_id:int) ~(out_id:int) : int32 =
  Int32.(add (of_int (in_id land 0xFFFF)) (shift_left (of_int (out_id land 0xFFFF)) 16))

(* Lower Conv3x3 op to ISA instructions *)
let lower_conv3x3 (input:tensor) (weight:tensor) (stride:int) (pad:int) (out:tensor) : insn list =
  let act_bits = bits_to_u8 input.bits in
  let wgt_bits = bits_to_u8 weight.bits in
  
  let h_in, w_in, ic_total = match input.shape with
    | [h; w; c] -> (h, w, c)
    | _ -> failwith "Conv3x3 input must be [H;W;C]"
  in
  
  let oc_total = match weight.shape with
    | [3; 3; oc; _] -> oc
    | _ -> failwith "Conv3x3 weight must be [3;3;OC;IC]"
  in
  
  let h_out = (h_in + 2*pad - 3) / stride + 1 in
  let w_out = (w_in + 2*pad - 3) / stride + 1 in
  
  (* Get tiles *)
  let tiles = Tiling.tile_conv3x3 input weight stride pad in
  
  (* Generate instruction for each tile *)
  List.map (fun (t:Tiling.tile) ->
    let h_in_tile, w_in_tile = Tiling.calc_input_size t.h_out t.w_out stride pad in
    let args = [
      make_mode_bits ~act_bits ~wgt_bits ~stride ~pad;
      make_shape0 ~h_in:h_in_tile ~w_in:w_in_tile;
      make_shape1 ~ic:t.ic_len ~oc:t.oc_len;
      make_tile0 ~y0:t.y0 ~x0:t.x0;
      make_tile1 ~oh:t.h_out ~ow:t.w_out;
      Int32.of_int t.wgt_bytes;
      Int32.of_int t.act_bytes;
      Int32.of_int t.out_bytes;
      make_meta ~in_id:0 ~out_id:0;
    ] in
    { opcode = Conv3x3; flags = flags_default; args }
  ) tiles

(* Lower Pool2D op *)
let lower_pool2d (input:tensor) (kind:pool_kind) (out:tensor) : insn list =
  let elem_bits = bits_to_u8 input.bits in
  let pool_kind_val = match kind with Max -> 0 | Avg -> 1 in
  
  let h_in, w_in, c = match input.shape with
    | [h; w; c] -> (h, w, c)
    | _ -> failwith "Pool2D input must be [H;W;C]"
  in
  
  let mode = Int32.(add (of_int elem_bits) (shift_left (of_int pool_kind_val) 8) 
                     |> add (shift_left (of_int 2) 16) |> add (shift_left (of_int 2) 24)) in
  let shape = Int32.(add (of_int (h_in land 0xFFFF)) (shift_left (of_int (w_in land 0xFFFF)) 16)) in
  let channels = Int32.of_int (c land 0xFFFF) in
  
  let numel_in = h_in * w_in * c in
  let numel_out = (h_in/2) * (w_in/2) * c in
  let act_bytes = (numel_in * elem_bits + 7) / 8 in
  let out_bytes = (numel_out * (bits_to_u8 out.bits) + 7) / 8 in
  
  let args = [
    mode;
    shape;
    channels;
    Int32.of_int act_bytes;
    Int32.of_int out_bytes;
    Int32.zero;
  ] in
  
  [{ opcode = Pool2D; flags = flags_default; args }]

(* Lower Unpool2D op *)
let lower_unpool2d (input:tensor) (kind:unpool_kind) (out:tensor) : insn list =
  let elem_bits = bits_to_u8 input.bits in
  let unpool_kind_val = match kind with NearestRepeat -> 0 in
  
  let h_in, w_in, c = match input.shape with
    | [h; w; c] -> (h, w, c)
    | _ -> failwith "Unpool2D input must be [H;W;C]"
  in
  
  let mode = Int32.(add (of_int elem_bits) (shift_left (of_int unpool_kind_val) 8)
                     |> add (shift_left (of_int 2) 16)) in
  let shape = Int32.(add (of_int (h_in land 0xFFFF)) (shift_left (of_int (w_in land 0xFFFF)) 16)) in
  let channels = Int32.of_int (c land 0xFFFF) in
  
  let numel_in = h_in * w_in * c in
  let numel_out = numel_in * 4 in
  let act_bytes = (numel_in * elem_bits + 7) / 8 in
  let out_bytes = (numel_out * (bits_to_u8 out.bits) + 7) / 8 in
  
  let args = [
    mode;
    shape;
    channels;
    Int32.of_int act_bytes;
    Int32.of_int out_bytes;
  ] in
  
  [{ opcode = Unpool2D; flags = flags_default; args }]

(* Lower ConcatC op *)
let lower_concatc (a:tensor) (b:tensor) (out:tensor) : insn list =
  let elem_bits = bits_to_u8 a.bits in
  
  let h, w, ca = match a.shape with
    | [h; w; c] -> (h, w, c)
    | _ -> failwith "ConcatC input a must be [H;W;C]"
  in
  
  let cb = match b.shape with
    | [h'; w'; c'] when h' = h && w' = w -> c'
    | _ -> failwith "ConcatC input b must have same H,W as a"
  in
  
  let mode = Int32.(add (of_int elem_bits) (shift_left (of_int 2) 8)) in
  let shape = Int32.(add (of_int (h land 0xFFFF)) (shift_left (of_int (w land 0xFFFF)) 16)) in
  let channels = Int32.(add (of_int (ca land 0xFFFF)) (shift_left (of_int (cb land 0xFFFF)) 16)) in
  
  let numel_a = h * w * ca in
  let numel_b = h * w * cb in
  let bytes_a = (numel_a * elem_bits + 7) / 8 in
  let bytes_b = (numel_b * elem_bits + 7) / 8 in
  let out_bytes = ((numel_a + numel_b) * elem_bits + 7) / 8 in
  
  let args = [
    mode;
    shape;
    channels;
    Int32.of_int bytes_a;
    Int32.of_int bytes_b;
    Int32.of_int out_bytes;
  ] in
  
  [{ opcode = ConcatC; flags = flags_default; args }]

(* Lower ActQuant op *)
let lower_actquant (input:tensor) (fn:act_fn) (out_bits:bits) (out:tensor) : insn list =
  let in_bits = bits_to_u8 input.bits in
  let out_bits_val = bits_to_u8 out_bits in
  let act_fn_val = match fn with Identity -> 0 | ReLU -> 1 in
  
  let h, w, c = match input.shape with
    | [h; w; c] -> (h, w, c)
    | _ -> failwith "ActQuant input must be [H;W;C]"
  in
  
  let mode = Int32.(add (of_int in_bits) (shift_left (of_int out_bits_val) 8)
                     |> add (shift_left (of_int act_fn_val) 16)) in
  let shape = Int32.(add (of_int (h land 0xFFFF)) (shift_left (of_int (w land 0xFFFF)) 16)) in
  let channels = Int32.of_int (c land 0xFFFF) in
  
  let numel = h * w * c in
  let in_bytes = (numel * in_bits + 7) / 8 in
  let out_bytes = (numel * out_bits_val + 7) / 8 in
  
  let args = [
    mode;
    shape;
    channels;
    Int32.of_int in_bytes;
    Int32.of_int out_bytes;
  ] in
  
  [{ opcode = ActQuant; flags = flags_default; args }]

(* Lower GEMM op *)
let lower_gemm (x:tensor) (w:tensor) (out:tensor) : insn list =
  let act_bits = bits_to_u8 x.bits in
  let wgt_bits = bits_to_u8 w.bits in
  
  let m, k = match x.shape with
    | [m; k] -> (m, k)
    | _ -> failwith "GEMM x must be [M;K]"
  in
  
  let n = match w.shape with
    | [n; k'] when k' = k -> n
    | _ -> failwith "GEMM w must be [N;K]"
  in
  
  (* Get tiles *)
  let tiles = Tiling.tile_gemm x w in
  
  List.map (fun (t:Tiling.gemm_tile) ->
    let mode = Int32.(add (of_int act_bits) (shift_left (of_int wgt_bits) 8)) in
    let shape0 = Int32.(add (of_int (t.m_len land 0xFFFF)) (shift_left (of_int (t.k_len land 0xFFFF)) 16)) in
    let shape1 = Int32.(add (of_int (t.n_len land 0xFFFF)) (shift_left (of_int 0) 16)) in
    
    let args = [
      mode;
      shape0;
      shape1;
      Int32.of_int t.w_bytes;
      Int32.of_int t.x_bytes;
      Int32.of_int t.out_bytes;
    ] in
    
    { opcode = Gemm; flags = flags_default; args }
  ) tiles

(* Lower a single AST op to ISA instructions *)
let lower_op (op:op) : insn list =
  match op with
  | Conv3x3 {input; weight; stride; pad; out} ->
      lower_conv3x3 input weight stride pad out
  | Pool2D {input; kind; out} ->
      lower_pool2d input kind out
  | Unpool2D {input; kind; out} ->
      lower_unpool2d input kind out
  | ConcatC {a; b; out} ->
      lower_concatc a b out
  | ActQuant {input; fn; out_bits; out} ->
      lower_actquant input fn out_bits out
  | Gemm {x; w; out} ->
      lower_gemm x w out
  | Store _ -> []
  | Comment _ -> []

(* Lower entire program *)
let lower_program (prog:program) : insn list =
  let insns = List.concat_map lower_op prog in
  insns @ [{ opcode = End; flags = 0; args = [] }]

(* Generate residual add via concat + conv1x1 *)
let lower_residual_add (a:tensor) (b:tensor) (out:tensor) : op list =
  (* Step 1: ConcatC a and b *)
  let concat_out_id = out.id ^ "_concat" in
  let ca = match a.shape with [_;_;c] -> c | _ -> 0 in
  let cb = match b.shape with [_;_;c] -> c | _ -> 0 in
  let h, w = match a.shape with [h;w;_] -> (h,w) | _ -> (0,0) in
  let concat_out = {
    id = concat_out_id;
    bits = a.bits;
    layout = HWC;
    shape = [h; w; ca+cb];
  } in
  
  (* Step 2: Conv1x1 using Conv3x3 with center weights *)
  (* Weight shape: [3;3;ca;ca+cb] but only center (1,1) used *)
  (* For residual add: W(1,1,oc,oc)=1, W(1,1,oc,oc+ca)=1 for oc in 0..ca-1 *)
  let weight_id = out.id ^ "_residual_w" in
  let weight = {
    id = weight_id;
    bits = B4;
    layout = HWC;
    shape = [3; 3; ca; ca+cb];
  } in
  
  let conv_out = { out with bits = B32 } in
  
  [
    ConcatC {a; b; out = concat_out};
    Conv3x3 {input = concat_out; weight; stride = 1; pad = 1; out = conv_out};
    ActQuant {input = conv_out; fn = Identity; out_bits = out.bits; out};
  ]
