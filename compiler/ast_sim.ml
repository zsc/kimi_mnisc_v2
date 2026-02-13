(* AST Simulator - bit-accurate reference implementation *)

open Ast_ir

(* ============================================================================
   MNISC-Q Encode/Decode
   ============================================================================ *)

(* Decode 2-bit code to signed int *)
let decode2 (code:int) : int =
  match code land 0x3 with
  | 0 -> -3   (* 00 *)
  | 1 -> -1   (* 01 *)
  | 2 -> 1    (* 10 *)
  | 3 -> 3    (* 11 *)
  | _ -> 0    (* unreachable *)

(* Encode signed int to 2-bit code (nearest odd) *)
let encode2 (value:int) : int =
  if value <= -2 then 0      (* -3 or less -> 00 *)
  else if value <= 0 then 1  (* -2,-1,0 -> 01 *)
  else if value <= 2 then 2  (* 1,2 -> 10 *)
  else 3                     (* 3+ -> 11 *)

(* Decode N-bit code (N=4,8,16) from packed byte buffer *)
let decodeN_bits (bits:bits) (code:int) : int =
  match bits with
  | B2 -> decode2 code
  | B4 ->
      (* 4-bit = 2 slices *)
      let s0 = decode2 (code land 0x3) in
      let s1 = decode2 ((code lsr 2) land 0x3) in
      s0 + (s1 lsl 2)
  | B8 ->
      (* 8-bit = 4 slices *)
      let s0 = decode2 (code land 0x3) in
      let s1 = decode2 ((code lsr 2) land 0x3) in
      let s2 = decode2 ((code lsr 4) land 0x3) in
      let s3 = decode2 ((code lsr 6) land 0x3) in
      s0 + (s1 lsl 2) + (s2 lsl 4) + (s3 lsl 6)
  | B16 ->
      (* 16-bit = 8 slices *)
      let result = ref 0 in
      for s = 0 to 7 do
        let slice = decode2 ((code lsr (2*s)) land 0x3) in
        result := !result + (slice lsl (2*s))
      done;
      !result
  | B32 -> code  (* B32 is plain int32 *)

(* Encode value to N-bit code *)
let encodeN_bits (bits:bits) (value:int) : int =
  match bits with
  | B2 -> encode2 value
  | B4 ->
      let s0 = encode2 (value land 0x3) in
      let s1 = encode2 ((value lsr 2) land 0x3) in
      s0 + (s1 lsl 2)
  | B8 ->
      let s0 = encode2 (value land 0x3) in
      let s1 = encode2 ((value lsr 2) land 0x3) in
      let s2 = encode2 ((value lsr 4) land 0x3) in
      let s3 = encode2 ((value lsr 6) land 0x3) in
      s0 + (s1 lsl 2) + (s2 lsl 4) + (s3 lsl 6)
  | B16 ->
      let result = ref 0 in
      for s = 0 to 7 do
        let slice = encode2 ((value lsr (2*s)) land 0x3) in
        result := !result + (slice lsl (2*s))
      done;
      !result
  | B32 -> value land 0xFFFFFFFF

(* Read element from packed byte buffer *)
let read_element (data:bytes) (idx:int) (bits:bits) : int =
  let elem_bits = match bits with B2 -> 2 | B4 -> 4 | B8 -> 8 | B16 -> 16 | B32 -> 32 in
  let total_bits = idx * elem_bits in
  let byte_idx = total_bits / 8 in
  let bit_offset = total_bits mod 8 in
  
  if elem_bits = 32 then
    (* Little endian 32-bit signed *)
    let b0 = Char.code (Bytes.get data byte_idx) in
    let b1 = Char.code (Bytes.get data (byte_idx+1)) in
    let b2 = Char.code (Bytes.get data (byte_idx+2)) in
    let b3 = Char.code (Bytes.get data (byte_idx+3)) in
    let code = b0 lor (b1 lsl 8) lor (b2 lsl 16) lor (b3 lsl 24) in
    (* Convert to signed 32-bit *)
    if code > 2147483647 then code - 4294967296 else code
  else if elem_bits = 16 then
    (* Little endian 16-bit *)
    let b0 = Char.code (Bytes.get data byte_idx) in
    let b1 = Char.code (Bytes.get data (byte_idx+1)) in
    let code = b0 + (b1 lsl 8) in
    decodeN_bits bits code
  else if elem_bits = 8 then
    decodeN_bits bits (Char.code (Bytes.get data byte_idx))
  else if elem_bits = 4 then
    let b = Char.code (Bytes.get data byte_idx) in
    let code = if bit_offset = 0 then b land 0xF else (b lsr 4) land 0xF in
    decodeN_bits bits code
  else (* elem_bits = 2 *)
    let b = Char.code (Bytes.get data byte_idx) in
    let code = (b lsr bit_offset) land 0x3 in
    decodeN_bits bits code

(* Write element to packed byte buffer *)
let write_element (data:bytes) (idx:int) (bits:bits) (value:int) : unit =
  let elem_bits = match bits with B2 -> 2 | B4 -> 4 | B8 -> 8 | B16 -> 16 | B32 -> 32 in
  let total_bits = idx * elem_bits in
  let byte_idx = total_bits / 8 in
  let bit_offset = total_bits mod 8 in
  
  if elem_bits = 32 then
    let code = value land 0xFFFFFFFF in
    Bytes.set data byte_idx (Char.chr (code land 0xFF));
    Bytes.set data (byte_idx+1) (Char.chr ((code lsr 8) land 0xFF));
    Bytes.set data (byte_idx+2) (Char.chr ((code lsr 16) land 0xFF));
    Bytes.set data (byte_idx+3) (Char.chr ((code lsr 24) land 0xFF))
  else if elem_bits = 16 then
    let code = encodeN_bits bits value in
    Bytes.set data byte_idx (Char.chr (code land 0xFF));
    Bytes.set data (byte_idx+1) (Char.chr ((code lsr 8) land 0xFF))
  else if elem_bits = 8 then
    let code = encodeN_bits bits value in
    Bytes.set data byte_idx (Char.chr code)
  else if elem_bits = 4 then
    let b = Char.code (Bytes.get data byte_idx) in
    let code = encodeN_bits bits value in
    let new_b = if bit_offset = 0 then
      (b land 0xF0) lor (code land 0xF)
    else
      (b land 0x0F) lor ((code land 0xF) lsl 4)
    in
    Bytes.set data byte_idx (Char.chr new_b)
  else (* elem_bits = 2 *)
    let b = Char.code (Bytes.get data byte_idx) in
    let code = encodeN_bits bits value in
    let new_b = (b land (lnot (0x3 lsl bit_offset))) lor ((code land 0x3) lsl bit_offset) in
    Bytes.set data byte_idx (Char.chr new_b)

(* ============================================================================
   Convolution Reference Implementation
   ============================================================================ *)

let conv3x3_ref 
    (packed_act:bytes) (act_tensor:tensor)
    (packed_wgt:bytes) (wgt_tensor:tensor)
    (stride:int) (pad:int) (shift1_en:bool) : bytes =
  
  let act_bits = match act_tensor.bits with B2 -> 2 | B4 -> 4 | B8 -> 8 | B16 -> 16 | B32 -> 32 in
  let wgt_bits = match wgt_tensor.bits with B2 -> 2 | B4 -> 4 | B8 -> 8 | B16 -> 16 | B32 -> 32 in
  
  let h_in, w_in, ic = match act_tensor.shape with [h;w;c] -> (h,w,c) | _ -> failwith "Bad act shape" in
  let oc = match wgt_tensor.shape with [3;3;oc;_] -> oc | _ -> failwith "Bad wgt shape" in
  
  let h_out = (h_in + 2*pad - 3) / stride + 1 in
  let w_out = (w_in + 2*pad - 3) / stride + 1 in
  
  let out_numel = h_out * w_out * oc in
  let out_bytes = Bytes.create (out_numel * 4) in  (* int32 output *)
  
  (* Helper to read activation with padding *)
  let read_act y x c =
    if y < 0 || y >= h_in || x < 0 || x >= w_in then 0
    else begin
      let idx = (y * w_in + x) * ic + c in
      read_element packed_act idx act_tensor.bits
    end
  in
  
  (* Helper to read weight: [kh][kw][oc][ic] *)
  let read_wgt kh kw oc_idx ic_idx =
    let idx = ((kh * 3 + kw) * oc + oc_idx) * ic + ic_idx in
    read_element packed_wgt idx wgt_tensor.bits
  in
  
  (* Compute convolution *)
  for oy = 0 to h_out - 1 do
    for ox = 0 to w_out - 1 do
      for oc_idx = 0 to oc - 1 do
        let sum = ref 0L in
        for ic_idx = 0 to ic - 1 do
          for kh = 0 to 2 do
            for kw = 0 to 2 do
              let iy = oy * stride + kh - pad in
              let ix = ox * stride + kw - pad in
              let a = read_act iy ix ic_idx in
              let w = read_wgt kh kw oc_idx ic_idx in
              sum := Int64.add !sum (Int64.of_int (a * w))
            done
          done
        done;
        
        (* Apply shift1 *)
        let result = if shift1_en then
          Int64.to_int (Int64.shift_right !sum 1)
        else
          Int64.to_int !sum
        in
        
        let out_idx = (oy * w_out + ox) * oc + oc_idx in
        let b0 = result land 0xFF in
        let b1 = (result lsr 8) land 0xFF in
        let b2 = (result lsr 16) land 0xFF in
        let b3 = (result lsr 24) land 0xFF in
        Bytes.set out_bytes (out_idx*4+0) (Char.chr b0);
        Bytes.set out_bytes (out_idx*4+1) (Char.chr b1);
        Bytes.set out_bytes (out_idx*4+2) (Char.chr b2);
        Bytes.set out_bytes (out_idx*4+3) (Char.chr b3)
      done
    done
  done;
  
  out_bytes

(* ============================================================================
   GEMM Reference Implementation
   ============================================================================ *)

let gemm_ref 
    (packed_x:bytes) (x_tensor:tensor)
    (packed_w:bytes) (w_tensor:tensor)
    (shift1_en:bool) : bytes =
  
  let act_bits = match x_tensor.bits with B2 -> 2 | B4 -> 4 | B8 -> 8 | B16 -> 16 | B32 -> 32 in
  let wgt_bits = match w_tensor.bits with B2 -> 2 | B4 -> 4 | B8 -> 8 | B16 -> 16 | B32 -> 32 in
  
  let m, k = match x_tensor.shape with [m;k] -> (m,k) | _ -> failwith "Bad x shape" in
  let n = match w_tensor.shape with [n;k'] when k' = k -> n | _ -> failwith "Bad w shape" in
  
  let out_bytes = Bytes.create (m * n * 4) in  (* int32 output *)
  
  for mi = 0 to m - 1 do
    for ni = 0 to n - 1 do
      let sum = ref 0L in
      for ki = 0 to k - 1 do
        let x_idx = mi * k + ki in
        let w_idx = ni * k + ki in
        let x_val = read_element packed_x x_idx x_tensor.bits in
        let w_val = read_element packed_w w_idx w_tensor.bits in
        sum := Int64.add !sum (Int64.of_int (x_val * w_val))
      done;
      
      let result = if shift1_en then
        Int64.to_int (Int64.shift_right !sum 1)
      else
        Int64.to_int !sum
      in
      
      let out_idx = mi * n + ni in
      let b0 = result land 0xFF in
      let b1 = (result lsr 8) land 0xFF in
      let b2 = (result lsr 16) land 0xFF in
      let b3 = (result lsr 24) land 0xFF in
      Bytes.set out_bytes (out_idx*4+0) (Char.chr b0);
      Bytes.set out_bytes (out_idx*4+1) (Char.chr b1);
      Bytes.set out_bytes (out_idx*4+2) (Char.chr b2);
      Bytes.set out_bytes (out_idx*4+3) (Char.chr b3)
    done
  done;
  
  out_bytes

(* ============================================================================
   Pooling Reference Implementation
   ============================================================================ *)

let pool2d_ref (packed_act:bytes) (act_tensor:tensor) (kind:pool_kind) : bytes =
  let elem_bits = match act_tensor.bits with B2 -> 2 | B4 -> 4 | B8 -> 8 | B16 -> 16 | B32 -> 32 in
  
  let h_in, w_in, c = match act_tensor.shape with [h;w;c] -> (h,w,c) | _ -> failwith "Bad shape" in
  let h_out = h_in / 2 in
  let w_out = w_in / 2 in
  
  let out_numel = h_out * w_out * c in
  let out_bits = max 8 elem_bits in
  let out_bytes = Bytes.create ((out_numel * out_bits + 7) / 8) in
  
  for oy = 0 to h_out - 1 do
    for ox = 0 to w_out - 1 do
      for ci = 0 to c - 1 do
        let vals = [|
          read_element packed_act (((oy*2+0) * w_in + ox*2+0) * c + ci) act_tensor.bits;
          read_element packed_act (((oy*2+0) * w_in + ox*2+1) * c + ci) act_tensor.bits;
          read_element packed_act (((oy*2+1) * w_in + ox*2+0) * c + ci) act_tensor.bits;
          read_element packed_act (((oy*2+1) * w_in + ox*2+1) * c + ci) act_tensor.bits;
        |] in
        
        let result = match kind with
          | Max -> Array.fold_left max (Array.get vals 0) vals
          | Avg -> (Array.fold_left (+) 0 vals) / 4
        in
        
        let out_idx = (oy * w_out + ox) * c + ci in
        write_element out_bytes out_idx act_tensor.bits result
      done
    done
  done;
  
  out_bytes

(* ============================================================================
   Unpooling Reference Implementation
   ============================================================================ *)

let unpool2d_ref (packed_act:bytes) (act_tensor:tensor) (kind:unpool_kind) : bytes =
  let elem_bits = match act_tensor.bits with B2 -> 2 | B4 -> 4 | B8 -> 8 | B16 -> 16 | B32 -> 32 in
  
  let h_in, w_in, c = match act_tensor.shape with [h;w;c] -> (h,w,c) | _ -> failwith "Bad shape" in
  let h_out = h_in * 2 in
  let w_out = w_in * 2 in
  
  let out_numel = h_out * w_out * c in
  let out_bytes = Bytes.create ((out_numel * elem_bits + 7) / 8) in
  
  match kind with
  | NearestRepeat ->
      for iy = 0 to h_in - 1 do
        for ix = 0 to w_in - 1 do
          for ci = 0 to c - 1 do
            let v = read_element packed_act ((iy * w_in + ix) * c + ci) act_tensor.bits in
            for dy = 0 to 1 do
              for dx = 0 to 1 do
                let oy = iy * 2 + dy in
                let ox = ix * 2 + dx in
                let out_idx = (oy * w_out + ox) * c + ci in
                write_element out_bytes out_idx act_tensor.bits v
              done
            done
          done
        done
      done;
      out_bytes

(* ============================================================================
   Concat Reference Implementation
   ============================================================================ *)

let concat_ref (packed_a:bytes) (a_tensor:tensor) (packed_b:bytes) (b_tensor:tensor) : bytes =
  let elem_bits = match a_tensor.bits with B2 -> 2 | B4 -> 4 | B8 -> 8 | B16 -> 16 | B32 -> 32 in
  
  let h, w, ca = match a_tensor.shape with [h;w;c] -> (h,w,c) | _ -> failwith "Bad a shape" in
  let cb = match b_tensor.shape with [_;_;c] -> c | _ -> failwith "Bad b shape" in
  
  let c_out = ca + cb in
  let out_numel = h * w * c_out in
  let out_bytes = Bytes.create ((out_numel * elem_bits + 7) / 8) in
  
  for y = 0 to h - 1 do
    for x = 0 to w - 1 do
      (* Copy from A *)
      for ci = 0 to ca - 1 do
        let src_idx = (y * w + x) * ca + ci in
        let dst_idx = (y * w + x) * c_out + ci in
        let v = read_element packed_a src_idx a_tensor.bits in
        write_element out_bytes dst_idx a_tensor.bits v
      done;
      (* Copy from B *)
      for ci = 0 to cb - 1 do
        let src_idx = (y * w + x) * cb + ci in
        let dst_idx = (y * w + x) * c_out + ca + ci in
        let v = read_element packed_b src_idx b_tensor.bits in
        write_element out_bytes dst_idx a_tensor.bits v
      done
    done
  done;
  
  out_bytes

(* ============================================================================
   Activation + Quantization Reference Implementation
   ============================================================================ *)

let act_quant_ref (packed_in:bytes) (in_tensor:tensor) (fn:act_fn) (out_bits:bits) : bytes =
  let in_elem_bits = match in_tensor.bits with B2 -> 2 | B4 -> 4 | B8 -> 8 | B16 -> 16 | B32 -> 32 in
  let out_elem_bits = match out_bits with B2 -> 2 | B4 -> 4 | B8 -> 8 | B16 -> 16 | B32 -> 32 in
  
  let h, w, c = match in_tensor.shape with [h;w;c] -> (h,w,c) | _ -> failwith "Bad shape" in
  let numel = h * w * c in
  let out_bytes = Bytes.create ((numel * out_elem_bits + 7) / 8) in
  
  (* Calculate representable range for output bits *)
  let max_val = (1 lsl out_elem_bits) - 1 in
  let min_val = -max_val in
  
  for i = 0 to numel - 1 do
    let v = read_element packed_in i in_tensor.bits in
    
    (* Apply activation function *)
    let v_act = match fn with
      | Identity -> v
      | ReLU -> if v < 0 then 0 else v
    in
    
    (* Requantize to nearest odd representable value *)
    (* Representable values are odd integers in range [-(2^N-1), +(2^N-1)] *)
    let v_quant = 
      (* Clamp to representable range first *)
      let v_clamped = max min_val (min max_val v_act) in
      (* Round to nearest odd - ties away from 0 *)
      let v_rounded = 
        if v_clamped land 1 = 1 then v_clamped  (* Already odd *)
        else if v_clamped >= 0 then v_clamped + 1  (* Even positive -> +1 *)
        else v_clamped - 1  (* Even negative -> -1 *)
      in
      (* Final clamp to ensure in range *)
      max min_val (min max_val v_rounded)
    in
    
    write_element out_bytes i out_bits v_quant
  done;
  
  out_bytes

(* ============================================================================
   Program Execution
   ============================================================================ *)

let run_program (prog:program) (inputs:(string, bytes) Hashtbl.t) : (string, bytes) Hashtbl.t =
  let outputs = Hashtbl.create 32 in
  let tensor_data = Hashtbl.create 32 in
  
  (* Copy inputs to tensor_data *)
  Hashtbl.iter (fun k v -> Hashtbl.add tensor_data k v) inputs;
  
  let get_tensor_data (t:tensor) : bytes =
    try Hashtbl.find tensor_data t.id
    with Not_found -> failwith (Printf.sprintf "Tensor %s not found" t.id)
  in
  
  let set_tensor_data (id:string) (data:bytes) : unit =
    Hashtbl.replace tensor_data id data
  in
  
  List.iter (fun op ->
    match op with
    | Conv3x3 {input; weight; stride; pad; out} ->
        let act_data = get_tensor_data input in
        let wgt_data = get_tensor_data weight in
        let out_data = conv3x3_ref act_data input wgt_data weight stride pad true in
        set_tensor_data out.id out_data
        
    | Gemm {x; w; out} ->
        let x_data = get_tensor_data x in
        let w_data = get_tensor_data w in
        let out_data = gemm_ref x_data x w_data w true in
        set_tensor_data out.id out_data
        
    | Pool2D {input; kind; out} ->
        let in_data = get_tensor_data input in
        let out_data = pool2d_ref in_data input kind in
        set_tensor_data out.id out_data
        
    | Unpool2D {input; kind; out} ->
        let in_data = get_tensor_data input in
        let out_data = unpool2d_ref in_data input kind in
        set_tensor_data out.id out_data
        
    | ConcatC {a; b; out} ->
        let a_data = get_tensor_data a in
        let b_data = get_tensor_data b in
        let out_data = concat_ref a_data a b_data b in
        set_tensor_data out.id out_data
        
    | ActQuant {input; fn; out_bits; out} ->
        let in_data = get_tensor_data input in
        let out_data = act_quant_ref in_data input fn out_bits in
        set_tensor_data out.id out_data
        
    | Store {input} ->
        let data = get_tensor_data input in
        Hashtbl.add outputs input.id (Bytes.copy data)
        
    | Comment _ -> ()
  ) prog;
  
  outputs
