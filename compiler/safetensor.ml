(* Safetensors format parser *)
(* Format: header_len (u64 LE) | header JSON | tensor data *)

type tensor_info = {
  shape : int list;
  dtype : string;
  offset : int;
  length : int;
}

type safetensor_file = {
  header : (string * tensor_info) list;
  data_offset : int;
  raw_bytes : bytes;
}

let parse_header (json_str:string) : (string * tensor_info) list =
  (* Simple JSON parsing for safetensors header *)
  (* Expected format: {"key": {"dtype": "U8", "shape": [1,2,3], "data_offsets": [0, 100]}, ...} *)
  let rec find_string_literal s pos =
    try
      let quote_pos = String.index_from s pos '"' in
      if quote_pos > 0 && s.[quote_pos-1] = '\\' then
        find_string_literal s (quote_pos+1)
      else
        Some quote_pos
    with Not_found -> None
  in
  
  let rec find_char s pos c =
    try Some (String.index_from s pos c) with Not_found -> None
  in
  
  let rec find_tensors s pos acc =
    if pos >= String.length s then acc
    else
      match find_string_literal s pos with
      | None -> acc
      | Some key_start ->
          match find_string_literal s (key_start+1) with
          | None -> acc
          | Some key_end ->
              let key = String.sub s (key_start+1) (key_end-key_start-1) in
              
              (* Find dtype - look for "dtype" pattern *)
              let dtype = ref "U8" in
              let shape = ref [] in
              let offset = ref 0 in
              let length = ref 0 in
              
              (* Look for shape array *)
              (match find_char s key_end '[' with
              | Some shape_start ->
                  (match find_char s (shape_start+1) ']' with
                  | Some shape_end ->
                      let shape_str = String.sub s (shape_start+1) (shape_end-shape_start-1) in
                      shape := if String.trim shape_str = "" then []
                               else shape_str |> String.split_on_char ',' 
                                    |> List.map String.trim 
                                    |> List.filter (fun x -> x <> "")
                                    |> List.map int_of_string
                  | None -> ())
              | None -> ());
              
              (* Look for data_offsets *)
              let offset_keyword = "data_offsets" in
              let rec find_keyword p =
                if p + String.length offset_keyword > String.length s then None
                else if String.sub s p (String.length offset_keyword) = offset_keyword then
                  Some p
                else find_keyword (p+1)
              in
              (match find_keyword key_end with
              | Some offset_pos ->
                  (match find_char s offset_pos '[' with
                  | Some arr_start ->
                      (match find_char s (arr_start+1) ']' with
                      | Some arr_end ->
                          let arr_str = String.sub s (arr_start+1) (arr_end-arr_start-1) in
                          let offsets = arr_str |> String.split_on_char ',' 
                                       |> List.map String.trim 
                                       |> List.filter (fun x -> x <> "") 
                                       |> List.map int_of_string in
                          if List.length offsets >= 2 then begin
                            offset := List.nth offsets 0;
                            length := List.nth offsets 1 - List.nth offsets 0
                          end
                      | None -> ())
                  | None -> ())
              | None -> ());
              
              let info = { shape = !shape; dtype = !dtype; offset = !offset; length = !length } in
              find_tensors s (key_end+1) ((key, info) :: acc)
  in
  
  find_tensors json_str 0 []

let load_file (path:string) : safetensor_file =
  let ic = open_in_bin path in
  let len = in_channel_length ic in
  let buf = really_input_string ic len in
  close_in ic;
  
  (* Read header length (u64 LE) *)
  let header_len = 
    let b0 = Char.code buf.[0] in
    let b1 = Char.code buf.[1] in
    let b2 = Char.code buf.[2] in
    let b3 = Char.code buf.[3] in
    let b4 = Char.code buf.[4] in
    let b5 = Char.code buf.[5] in
    let b6 = Char.code buf.[6] in
    let b7 = Char.code buf.[7] in
    Int64.to_int (Int64.logor 
      (Int64.logor (Int64.logor (Int64.of_int b0) (Int64.shift_left (Int64.of_int b1) 8))
                   (Int64.logor (Int64.shift_left (Int64.of_int b2) 16) (Int64.shift_left (Int64.of_int b3) 24)))
      (Int64.logor (Int64.logor (Int64.shift_left (Int64.of_int b4) 32) (Int64.shift_left (Int64.of_int b5) 40))
                   (Int64.logor (Int64.shift_left (Int64.of_int b6) 48) (Int64.shift_left (Int64.of_int b7) 56))))
  in
  
  (* Read header JSON *)
  let header_json = String.sub buf 8 header_len in
  let header = parse_header header_json in
  
  (* Raw bytes start after header_len field + header *)
  let data_offset = 8 + header_len in
  let raw_bytes = Bytes.of_string (String.sub buf data_offset (len - data_offset)) in
  
  { header; data_offset; raw_bytes }

let load_tensor (path:string) (key:string) : Ast_ir.tensor * bytes =
  let st = load_file path in
  let info = try List.assoc key st.header with Not_found -> 
    failwith (Printf.sprintf "Key %s not found in safetensor file" key)
  in
  
  (* Create tensor descriptor *)
  let bits = match info.dtype with
    | "U8" | "I8" -> Ast_ir.B8
    | "U4" | "I4" -> Ast_ir.B4
    | "U2" | "I2" -> Ast_ir.B2
    | "I32" -> Ast_ir.B32
    | "I16" -> Ast_ir.B16
    | _ -> Ast_ir.B8
  in
  
  let tensor = {
    Ast_ir.id = key;
    bits = bits;
    layout = Ast_ir.HWC;
    shape = info.shape;
  } in
  
  (* Extract tensor data *)
  let data = Bytes.sub st.raw_bytes info.offset info.length in
  (tensor, data)

let list_keys (path:string) : string list =
  let st = load_file path in
  List.map fst st.header
