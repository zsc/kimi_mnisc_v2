type opcode =
  | Nop | End | MetaTensorDef | MetaBar
  | Conv3x3 | Pool2D | Unpool2D | ConcatC | ActQuant | Gemm

type insn = { opcode: opcode; flags:int; args:int32 list }

let opcode_to_u8 = function
  | Nop -> 0x00 | End -> 0x01
  | MetaTensorDef -> 0x10 | MetaBar -> 0x11
  | Conv3x3 -> 0x20 | Pool2D -> 0x21 | Unpool2D -> 0x22
  | ConcatC -> 0x23 | ActQuant -> 0x24 | Gemm -> 0x25

let encode_header ~(opc:int) ~(flags:int) : int32 =
  let opc = opc land 0xFF in
  let flags = flags land 0xFF in
  Int32.of_int (opc lor (flags lsl 8))

let encode_program_u32 (p:insn list) : int32 array =
  let words =
    List.concat_map (fun i ->
      let h = encode_header ~opc:(opcode_to_u8 i.opcode) ~flags:i.flags in
      h :: i.args
    ) p
  in
  Array.of_list words
