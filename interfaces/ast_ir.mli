type bits = B2 | B4 | B8 | B16 | B32
type layout = HWC | MK
type tensor_id = string

type tensor = {
  id     : tensor_id;
  bits   : bits;
  layout : layout;
  shape  : int list;
}

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
val infer_shapes : program -> (tensor_id, tensor) Hashtbl.t
