type opcode =
  | Nop | End | MetaTensorDef | MetaBar
  | Conv3x3 | Pool2D | Unpool2D | ConcatC | ActQuant | Gemm

type insn = {
  opcode : opcode;
  flags  : int;
  args   : int32 list;
}

val encode_program_u32 : insn list -> int32 array
