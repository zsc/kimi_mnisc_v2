(* Tiling logic for MNISC Conv3x3 and GEMM *)
(* Hardware constraints: MAX_H, MAX_W, MAX_IC=16, MAX_OC=16 *)

open Ast_ir

type tile = {
  y0 : int;
  x0 : int;
  h_out : int;
  w_out : int;
  ic0 : int;
  ic_len : int;
  oc0 : int;
  oc_len : int;
  act_bytes : int;
  wgt_bytes : int;
  out_bytes : int;
}

(* Hardware constraints *)
let max_h = 64
let max_w = 64
let max_ic = 16
let max_oc = 16
let wbuf_bytes_max = 65536

let bits_to_int = function
  | B2 -> 2 | B4 -> 4 | B8 -> 8 | B16 -> 16 | B32 -> 32

let calc_act_bytes (h:int) (w:int) (ic:int) (act_bits:int) : int =
  let numel = h * w * ic in
  (numel * act_bits + 7) / 8

let calc_wgt_bytes (kh:int) (kw:int) (ic:int) (oc:int) (wgt_bits:int) : int =
  let numel = kh * kw * ic * oc in
  (numel * wgt_bits + 7) / 8

let calc_out_bytes (h:int) (w:int) (oc:int) : int =
  h * w * oc * 4  (* int32 output *)

(* Calculate input spatial size for given output size, stride, pad *)
let calc_input_size (oh:int) (ow:int) (stride:int) (pad:int) : int * int =
  (* With pad=1: input_H = output_H * stride + 2 (due to 3x3 kernel) *)
  (* With pad=0: input_H = output_H * stride + 2 *)
  let ih = oh * stride + 2 in
  let iw = ow * stride + 2 in
  (ih, iw)

(* Generate tiles for Conv3x3 *)
let tile_conv3x3 (input:tensor) (weight:tensor) (stride:int) (pad:int) : tile list =
  let act_bits = bits_to_int input.bits in
  let wgt_bits = bits_to_int weight.bits in
  
  (* Extract dimensions from tensors *)
  let h_in, w_in, ic_total = match input.shape with
    | [h; w; c] -> (h, w, c)
    | _ -> failwith "Conv3x3 input must have shape [H;W;C]"
  in
  
  let oc_total = match weight.shape with
    | [3; 3; oc; _ic] -> oc
    | _ -> failwith "Conv3x3 weight must have shape [3;3;OC;IC]"
  in
  
  (* Calculate output dimensions *)
  let h_out = (h_in + 2*pad - 3) / stride + 1 in
  let w_out = (w_in + 2*pad - 3) / stride + 1 in
  
  let tiles = ref [] in
  
  (* Tile over OC first (weight buffer constraint) *)
  let oc_tiles = (oc_total + max_oc - 1) / max_oc in
  let ic_tiles = (ic_total + max_ic - 1) / max_ic in
  
  for oc_tile = 0 to oc_tiles - 1 do
    let oc0 = oc_tile * max_oc in
    let oc_len = min max_oc (oc_total - oc0) in
    
    for ic_tile = 0 to ic_tiles - 1 do
      let ic0 = ic_tile * max_ic in
      let ic_len = min max_ic (ic_total - ic0) in
      
      (* For simplicity, use full spatial dimensions or tile if needed *)
      (* If output is small enough, use single tile *)
      let h_tiles = if h_out <= max_h then 1 else (h_out + max_h - 1) / max_h in
      let w_tiles = if w_out <= max_w then 1 else (w_out + max_w - 1) / max_w in
      
      for y_tile = 0 to h_tiles - 1 do
        for x_tile = 0 to w_tiles - 1 do
          let y0 = y_tile * max_h in
          let x0 = x_tile * max_w in
          let h_tile_out = min max_h (h_out - y0) in
          let w_tile_out = min max_w (w_out - x0) in
          
          (* Calculate input region needed *)
          let h_tile_in, w_tile_in = calc_input_size h_tile_out w_tile_out stride pad in
          
          (* Calculate bytes *)
          let act_b = calc_act_bytes h_tile_in w_tile_in ic_len act_bits in
          let wgt_b = calc_wgt_bytes 3 3 ic_len oc_len wgt_bits in
          let out_b = calc_out_bytes h_tile_out w_tile_out oc_len in
          
          let tile = {
            y0; x0;
            h_out = h_tile_out; w_out = w_tile_out;
            ic0; ic_len;
            oc0; oc_len;
            act_bytes = act_b;
            wgt_bytes = wgt_b;
            out_bytes = out_b;
          } in
          
          tiles := tile :: !tiles
        done
      done
    done
  done;
  
  List.rev !tiles

(* Generate tiles for GEMM *)
type gemm_tile = {
  m0 : int; m_len : int;
  n0 : int; n_len : int;
  k0 : int; k_len : int;
  x_bytes : int;
  w_bytes : int;
  out_bytes : int;
}

let tile_gemm (x:tensor) (w:tensor) : gemm_tile list =
  let act_bits = bits_to_int x.bits in
  let wgt_bits = bits_to_int w.bits in
  
  let m, k = match x.shape with
    | [m; k] -> (m, k)
    | _ -> failwith "GEMM x must have shape [M;K]"
  in
  
  let n = match w.shape with
    | [n; k'] when k' = k -> n
    | _ -> failwith "GEMM w must have shape [N;K]"
  in
  
  let tiles = ref [] in
  
  let m_tile_size = 16 in
  let n_tile_size = 16 in
  let k_tile_size = 16 in
  
  let m_tiles = (m + m_tile_size - 1) / m_tile_size in
  let n_tiles = (n + n_tile_size - 1) / n_tile_size in
  let k_tiles = (k + k_tile_size - 1) / k_tile_size in
  
  for mt = 0 to m_tiles - 1 do
    for nt = 0 to n_tiles - 1 do
      for kt = 0 to k_tiles - 1 do
        let m0 = mt * m_tile_size in
        let n0 = nt * n_tile_size in
        let k0 = kt * k_tile_size in
        let m_len = min m_tile_size (m - m0) in
        let n_len = min n_tile_size (n - n0) in
        let k_len = min k_tile_size (k - k0) in
        
        let x_b = calc_act_bytes m_len k_len 1 act_bits in
        let w_b = calc_act_bytes n_len k_len 1 wgt_bits in
        let out_b = calc_out_bytes m_len n_len 1 in
        
        let tile = {
          m0; m_len; n0; n_len; k0; k_len;
          x_bytes = x_b;
          w_bytes = w_b;
          out_bytes = out_b;
        } in
        
        tiles := tile :: !tiles
      done
    done
  done;
  
  List.rev !tiles
