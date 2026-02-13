type bits = B2 | B4 | B8 | B16 | B32
type layout = HWC | MK
type tensor_id = string

type tensor = { id: tensor_id; bits: bits; layout: layout; shape: int list }

type act_fn = Identity | ReLU
type pool_kind = Max | Avg
type unpool_kind = NearestRepeat

type op =
  | Conv3x3 of { input:tensor; weight:tensor; stride:int; pad:int; out:tensor }
  | Gemm of { x:tensor; w:tensor; out:tensor }
  | Pool2D of { input:tensor; kind:pool_kind; out:tensor }
  | Unpool2D of { input:tensor; kind:unpool_kind; out:tensor }
  | ConcatC of { a:tensor; b:tensor; out:tensor }
  | ActQuant of { input:tensor; fn:act_fn; out_bits:bits; out:tensor }
  | Store of { input:tensor }
  | Comment of string

type program = op list

let bits_to_int = function
  | B2 -> 2 | B4 -> 4 | B8 -> 8 | B16 -> 16 | B32 -> 32

let tensor_numel (t:tensor) : int =
  List.fold_left ( * ) 1 t.shape

let tensor_bytes (t:tensor) : int =
  let numel = tensor_numel t in
  let bits = bits_to_int t.bits in
  (numel * bits + 7) / 8

let calc_conv_out_shape (h:int) (w:int) (stride:int) (pad:int) : int * int =
  let oh = (h + 2*pad - 3) / stride + 1 in
  let ow = (w + 2*pad - 3) / stride + 1 in
  (oh, ow)

let calc_pool_out_shape (h:int) (w:int) : int * int =
  (h / 2, w / 2)

let calc_unpool_out_shape (h:int) (w:int) : int * int =
  (h * 2, w * 2)

let infer_shapes (p:program) : (tensor_id, tensor) Hashtbl.t =
  let tbl = Hashtbl.create 32 in
  let add_tensor t = Hashtbl.add tbl t.id t in
  
  List.iter (fun op ->
    match op with
    | Conv3x3 {input; weight; stride; pad; out} ->
        add_tensor input;
        add_tensor weight;
        add_tensor out
    | Gemm {x; w; out} ->
        add_tensor x;
        add_tensor w;
        add_tensor out
    | Pool2D {input; out; _} ->
        add_tensor input;
        add_tensor out
    | Unpool2D {input; out; _} ->
        add_tensor input;
        add_tensor out
    | ConcatC {a; b; out} ->
        add_tensor a;
        add_tensor b;
        add_tensor out
    | ActQuant {input; out; _} ->
        add_tensor input;
        add_tensor out
    | Store {input} ->
        add_tensor input
    | Comment _ -> ()
  ) p;
  tbl
