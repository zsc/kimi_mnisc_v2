(* MNISC Compiler Main *)

open Ast_ir

(* Don't open Eu_isa to avoid name conflicts with Ast_ir constructors *)

(* Default U-Net topology for reference *)
let build_unet_program () : program =
  (* Helper to create tensors *)
  let tensor id bits shape = { id; bits; layout = HWC; shape } in
  
  (* Input: [32,32,1] *)
  let x0 = tensor "input.x" B8 [32; 32; 1] in
  
  (* Encoder *)
  let conv1a_out = tensor "conv1a_out" B32 [32; 32; 16] in
  let conv1a = Conv3x3 {
    input = x0;
    weight = tensor "conv1a.weight" B4 [3; 3; 16; 1];
    stride = 1; pad = 1;
    out = conv1a_out
  } in
  let relu1a_out = tensor "relu1a_out" B8 [32; 32; 16] in
  let relu1a = ActQuant { input = conv1a_out; fn = ReLU; out_bits = B8; out = relu1a_out } in
  
  let conv1b_out = tensor "conv1b_out" B32 [32; 32; 16] in
  let conv1b = Conv3x3 {
    input = relu1a_out;
    weight = tensor "conv1b.weight" B4 [3; 3; 16; 16];
    stride = 1; pad = 1;
    out = conv1b_out
  } in
  let relu1b_out = tensor "relu1b_out" B8 [32; 32; 16] in
  let relu1b = ActQuant { input = conv1b_out; fn = ReLU; out_bits = B8; out = relu1b_out } in
  
  let skip1 = relu1b_out in  (* For concat later *)
  
  let pool1_out = tensor "pool1_out" B8 [16; 16; 16] in
  let pool1 = Pool2D { input = relu1b_out; kind = Max; out = pool1_out } in
  
  let conv2a_out = tensor "conv2a_out" B32 [16; 16; 32] in
  let conv2a = Conv3x3 {
    input = pool1_out;
    weight = tensor "conv2a.weight" B4 [3; 3; 32; 16];
    stride = 1; pad = 1;
    out = conv2a_out
  } in
  let relu2a_out = tensor "relu2a_out" B8 [16; 16; 32] in
  let relu2a = ActQuant { input = conv2a_out; fn = ReLU; out_bits = B8; out = relu2a_out } in
  
  let conv2b_out = tensor "conv2b_out" B32 [16; 16; 32] in
  let conv2b = Conv3x3 {
    input = relu2a_out;
    weight = tensor "conv2b.weight" B4 [3; 3; 32; 32];
    stride = 1; pad = 1;
    out = conv2b_out
  } in
  let relu2b_out = tensor "relu2b_out" B8 [16; 16; 32] in
  let relu2b = ActQuant { input = conv2b_out; fn = ReLU; out_bits = B8; out = relu2b_out } in
  
  let skip2 = relu2b_out in
  
  let pool2_out = tensor "pool2_out" B8 [8; 8; 32] in
  let pool2 = Pool2D { input = relu2b_out; kind = Max; out = pool2_out } in
  
  (* Bottleneck *)
  let conv3a_out = tensor "conv3a_out" B32 [8; 8; 64] in
  let conv3a = Conv3x3 {
    input = pool2_out;
    weight = tensor "conv3a.weight" B4 [3; 3; 64; 32];
    stride = 1; pad = 1;
    out = conv3a_out
  } in
  let relu3a_out = tensor "relu3a_out" B8 [8; 8; 64] in
  let relu3a = ActQuant { input = conv3a_out; fn = ReLU; out_bits = B8; out = relu3a_out } in
  
  let conv3b_out = tensor "conv3b_out" B32 [8; 8; 64] in
  let conv3b = Conv3x3 {
    input = relu3a_out;
    weight = tensor "conv3b.weight" B4 [3; 3; 64; 64];
    stride = 1; pad = 1;
    out = conv3b_out
  } in
  let relu3b_out = tensor "relu3b_out" B8 [8; 8; 64] in
  let relu3b = ActQuant { input = conv3b_out; fn = ReLU; out_bits = B8; out = relu3b_out } in
  
  (* Decoder *)
  let unpool2_out = tensor "unpool2_out" B8 [16; 16; 64] in
  let unpool2 = Unpool2D { input = relu3b_out; kind = NearestRepeat; out = unpool2_out } in
  
  let concat2_out = tensor "concat2_out" B8 [16; 16; 96] in
  let concat2 = ConcatC { a = skip2; b = unpool2_out; out = concat2_out } in
  
  let conv4a_out = tensor "conv4a_out" B32 [16; 16; 32] in
  let conv4a = Conv3x3 {
    input = concat2_out;
    weight = tensor "conv4a.weight" B4 [3; 3; 32; 96];
    stride = 1; pad = 1;
    out = conv4a_out
  } in
  let relu4a_out = tensor "relu4a_out" B8 [16; 16; 32] in
  let relu4a = ActQuant { input = conv4a_out; fn = ReLU; out_bits = B8; out = relu4a_out } in
  
  let conv4b_out = tensor "conv4b_out" B32 [16; 16; 32] in
  let conv4b = Conv3x3 {
    input = relu4a_out;
    weight = tensor "conv4b.weight" B4 [3; 3; 32; 32];
    stride = 1; pad = 1;
    out = conv4b_out
  } in
  let relu4b_out = tensor "relu4b_out" B8 [16; 16; 32] in
  let relu4b = ActQuant { input = conv4b_out; fn = ReLU; out_bits = B8; out = relu4b_out } in
  
  let unpool1_out = tensor "unpool1_out" B8 [32; 32; 32] in
  let unpool1 = Unpool2D { input = relu4b_out; kind = NearestRepeat; out = unpool1_out } in
  
  let concat1_out = tensor "concat1_out" B8 [32; 32; 48] in
  let concat1 = ConcatC { a = skip1; b = unpool1_out; out = concat1_out } in
  
  let conv5a_out = tensor "conv5a_out" B32 [32; 32; 16] in
  let conv5a = Conv3x3 {
    input = concat1_out;
    weight = tensor "conv5a.weight" B4 [3; 3; 16; 48];
    stride = 1; pad = 1;
    out = conv5a_out
  } in
  let relu5a_out = tensor "relu5a_out" B8 [32; 32; 16] in
  let relu5a = ActQuant { input = conv5a_out; fn = ReLU; out_bits = B8; out = relu5a_out } in
  
  let conv5b_out = tensor "conv5b_out" B32 [32; 32; 16] in
  let conv5b = Conv3x3 {
    input = relu5a_out;
    weight = tensor "conv5b.weight" B4 [3; 3; 16; 16];
    stride = 1; pad = 1;
    out = conv5b_out
  } in
  let relu5b_out = tensor "relu5b_out" B8 [32; 32; 16] in
  let relu5b = ActQuant { input = conv5b_out; fn = ReLU; out_bits = B8; out = relu5b_out } in
  
  (* Output conv 1x1 *)
  let out_conv_out = tensor "out_conv_out" B32 [32; 32; 1] in
  let out_conv = Conv3x3 {
    input = relu5b_out;
    weight = tensor "out_conv.weight" B4 [3; 3; 1; 16];
    stride = 1; pad = 1;
    out = out_conv_out
  } in
  let output = tensor "output" B8 [32; 32; 1] in
  let out_quant = ActQuant { input = out_conv_out; fn = Identity; out_bits = B8; out = output } in
  
  (* Build program *)
  [
    Comment "=== Encoder ===";
    conv1a; relu1a;
    conv1b; relu1b;
    pool1;
    conv2a; relu2a;
    conv2b; relu2b;
    pool2;
    Comment "=== Bottleneck ===";
    conv3a; relu3a;
    conv3b; relu3b;
    Comment "=== Decoder ===";
    unpool2; concat2;
    conv4a; relu4a;
    conv4b; relu4b;
    unpool1; concat1;
    conv5a; relu5a;
    conv5b; relu5b;
    Comment "=== Output ===";
    out_conv; out_quant;
    Store {input = output};
  ]

(* Generate random weights and save to safetensors file *)
let generate_random_weights (path:string) : unit =
  let open Safetensor in
  
  (* Weight specifications: (key, shape, dtype) *)
  let weights = [
    ("conv1a.weight", [3; 3; 16; 1], "U8");
    ("conv1b.weight", [3; 3; 16; 16], "U8");
    ("conv2a.weight", [3; 3; 32; 16], "U8");
    ("conv2b.weight", [3; 3; 32; 32], "U8");
    ("conv3a.weight", [3; 3; 64; 32], "U8");
    ("conv3b.weight", [3; 3; 64; 64], "U8");
    ("conv4a.weight", [3; 3; 32; 96], "U8");
    ("conv4b.weight", [3; 3; 32; 32], "U8");
    ("conv5a.weight", [3; 3; 16; 48], "U8");
    ("conv5b.weight", [3; 3; 16; 16], "U8");
    ("out_conv.weight", [3; 3; 1; 16], "U8");
  ] in
  
  (* Generate random data *)
  let total_bytes = ref 0 in
  let weight_data = List.map (fun (key, shape, dtype) ->
    let numel = List.fold_left ( * ) 1 shape in
    (* int4 packed: 2 values per byte *)
    let bytes_needed = (numel + 1) / 2 in
    let data = Bytes.create bytes_needed in
    for i = 0 to bytes_needed - 1 do
      let b = Random.int 256 in
      Bytes.set data i (Char.chr b)
    done;
    total_bytes := !total_bytes + bytes_needed;
    ((key, shape, dtype, bytes_needed), data)
  ) weights in
  
  (* Build header JSON *)
  let header_parts = ref [] in
  let offset = ref 0 in
  List.iter (fun ((key, shape, dtype, len), _) ->
    let shape_str = String.concat ", " (List.map string_of_int shape) in
    let entry = Printf.sprintf "\"%s\": {\"dtype\": \"%s\", \"shape\": [%s], \"data_offsets\": [%d, %d]}"
      key dtype shape_str !offset (!offset + len) in
    header_parts := entry :: !header_parts;
    offset := !offset + len
  ) weight_data;
  
  let header_json = "{" ^ (String.concat ", " (List.rev !header_parts)) ^ "}" in
  let header_len = String.length header_json in
  
  (* Write file *)
  let oc = open_out_bin path in
  (* Write header length as u64 LE *)
  output_char oc (Char.chr (header_len land 0xFF));
  output_char oc (Char.chr ((header_len lsr 8) land 0xFF));
  output_char oc (Char.chr ((header_len lsr 16) land 0xFF));
  output_char oc (Char.chr ((header_len lsr 24) land 0xFF));
  output_char oc '\x00'; output_char oc '\x00'; output_char oc '\x00'; output_char oc '\x00';
  (* Write header JSON *)
  output_string oc header_json;
  (* Write tensor data *)
  List.iter (fun (_, data) ->
    output_bytes oc data
  ) weight_data;
  close_out oc;
  
  Printf.printf "Generated weights file: %s\n" path

(* Generate random input *)
let generate_random_input (path:string) : unit =
  let numel = 32 * 32 * 1 in  (* [32,32,1] *)
  let data = Bytes.create numel in
  for i = 0 to numel - 1 do
    Bytes.set data i (Char.chr (Random.int 256))
  done;
  
  let header_json = "{\"input.x\": {\"dtype\": \"U8\", \"shape\": [32, 32, 1], \"data_offsets\": [0, 1024]}}" in
  let header_len = String.length header_json in
  
  let oc = open_out_bin path in
  output_char oc (Char.chr (header_len land 0xFF));
  output_char oc (Char.chr ((header_len lsr 8) land 0xFF));
  output_char oc (Char.chr ((header_len lsr 16) land 0xFF));
  output_char oc (Char.chr ((header_len lsr 24) land 0xFF));
  output_char oc '\x00'; output_char oc '\x00'; output_char oc '\x00'; output_char oc '\x00';
  output_string oc header_json;
  output_bytes oc data;
  close_out oc;
  
  Printf.printf "Generated input file: %s\n" path

(* Save program binary *)
let save_program_binary (path:string) (insns:int32 array) : unit =
  let oc = open_out_bin path in
  Array.iter (fun word ->
    let b0 = Int32.to_int (Int32.logand word 0xFFl) in
    let b1 = Int32.to_int (Int32.logand (Int32.shift_right_logical word 8) 0xFFl) in
    let b2 = Int32.to_int (Int32.logand (Int32.shift_right_logical word 16) 0xFFl) in
    let b3 = Int32.to_int (Int32.logand (Int32.shift_right_logical word 24) 0xFFl) in
    output_char oc (Char.chr b0);
    output_char oc (Char.chr b1);
    output_char oc (Char.chr b2);
    output_char oc (Char.chr b3)
  ) insns;
  close_out oc;
  Printf.printf "Saved program binary: %s (%d instructions)\n" path (Array.length insns)

(* Save program metadata as JSON *)
let save_program_meta (path:string) (insns:Eu_isa.insn list) : unit =
  let json_parts = ref [] in
  let add_line s = json_parts := s :: !json_parts in
  
  add_line "{";
  add_line "  \"instructions\": [";
  
  let insn_count = List.length insns in
  List.iteri (fun i insn ->
    let opc_str = match insn.Eu_isa.opcode with
      | Nop -> "Nop" | End -> "End"
      | MetaTensorDef -> "MetaTensorDef" | MetaBar -> "MetaBar"
      | Conv3x3 -> "Conv3x3" | Pool2D -> "Pool2D" 
      | Unpool2D -> "Unpool2D" | ConcatC -> "ConcatC"
      | ActQuant -> "ActQuant" | Gemm -> "Gemm"
    in
    let args_str = String.concat ", " (List.map (fun a -> Int32.to_string a) insn.Eu_isa.args) in
    let line = Printf.sprintf "    {\"opcode\": \"%s\", \"flags\": %d, \"args\": [%s]}%s"
      opc_str insn.Eu_isa.flags args_str (if i < insn_count - 1 then "," else "")
    in
    add_line line
  ) insns;
  
  add_line "  ]";
  add_line "}";
  
  let oc = open_out path in
  List.iter (fun s -> output_string oc s; output_char oc '\n') (List.rev !json_parts);
  close_out oc;
  Printf.printf "Saved program metadata: %s\n" path

(* Main entry point *)
let () =
  Random.self_init ();
  
  (* Parse command line *)
  let args = Sys.argv |> Array.to_list |> List.tl in
  
  let model_path = ref "model.safetensors" in
  let input_path = ref "input.safetensors" in
  let output_bin = ref "program.bin" in
  let output_meta = ref "program_meta.json" in
  let gen_weights = ref false in
  let run_sim = ref false in
  
  let rec parse_args = function
    | [] -> ()
    | "--model" :: v :: rest -> model_path := v; parse_args rest
    | "--input" :: v :: rest -> input_path := v; parse_args rest
    | "--output-bin" :: v :: rest -> output_bin := v; parse_args rest
    | "--output-meta" :: v :: rest -> output_meta := v; parse_args rest
    | "--gen-weights" :: rest -> gen_weights := true; parse_args rest
    | "--run-sim" :: rest -> run_sim := true; parse_args rest
    | _ :: rest -> parse_args rest
  in
  parse_args args;
  
  (* Generate weights if requested *)
  if !gen_weights then begin
    generate_random_weights !model_path;
    generate_random_input !input_path
  end;
  
  (* Build AST program *)
  Printf.printf "Building AST program...\n";
  let program = build_unet_program () in
  
  (* Run shape inference *)
  let shapes = Ast_ir.infer_shapes program in
  Printf.printf "Inferred %d tensors\n" (Hashtbl.length shapes);
  
  (* Lower to ISA *)
  Printf.printf "Lowering to EU ISA...\n";
  let insns = Lower.lower_program program in
  Printf.printf "Generated %d instructions\n" (List.length insns);
  
  (* Encode to binary *)
  let encoded = Eu_isa.encode_program_u32 insns in
  save_program_binary !output_bin encoded;
  save_program_meta !output_meta insns;
  
  (* Run AST simulation if requested *)
  if !run_sim then begin
    Printf.printf "Running AST simulation...\n";
    (* Load inputs *)
    let inputs = Hashtbl.create 16 in
    
    (* Load model weights *)
    let weight_keys = [
      "conv1a.weight"; "conv1b.weight"; "conv2a.weight"; "conv2b.weight";
      "conv3a.weight"; "conv3b.weight"; "conv4a.weight"; "conv4b.weight";
      "conv5a.weight"; "conv5b.weight"; "out_conv.weight"
    ] in
    List.iter (fun key ->
      try
        let (_tensor, data) = Safetensor.load_tensor !model_path key in
        Hashtbl.add inputs key data
      with e ->
        Printf.printf "Warning: could not load weight %s: %s\n" key (Printexc.to_string e)
    ) weight_keys;
    
    (* Load input tensor *)
    try
      let (_tensor, data) = Safetensor.load_tensor !input_path "input.x" in
      Hashtbl.add inputs "input.x" data
    with e ->
      Printf.printf "Warning: could not load input: %s\n" (Printexc.to_string e);
      (* Generate dummy input *)
      Hashtbl.add inputs "input.x" (Bytes.create 1024)
    ;
    
    (* Run simulation *)
    let outputs = Ast_sim.run_program program inputs in
    Printf.printf "Simulation complete. Stored %d output tensors.\n" (Hashtbl.length outputs);
    
    (* Print output info *)
    Hashtbl.iter (fun key data ->
      Printf.printf "  %s: %d bytes\n" key (Bytes.length data)
    ) outputs
  end;
  
  Printf.printf "Done.\n"
