import struct
import json

# Create program.bin with OPC_END (0x01) instruction
with open('program.bin', 'wb') as f:
    # OPC_END = 0x00000001 (header: opcode=0x01, flags=0x00, reserved=0x0000)
    f.write(struct.pack('<I', 0x00000001))

# Create program_meta.json
meta = {
    "instructions": [
        {"opcode": "OPC_END"}
    ]
}
with open('program_meta.json', 'w') as f:
    json.dump(meta, f, indent=2)

# Create valid empty safetensors files
def create_empty_safetensors(filename):
    header = json.dumps({})
    header_bytes = header.encode('utf-8')
    # Pad to 8-byte boundary
    padding = (8 - len(header_bytes) % 8) % 8
    header_bytes += b' ' * padding
    
    with open(filename, 'wb') as f:
        # Write header size (uint64)
        f.write(struct.pack('<Q', len(header_bytes)))
        # Write header
        f.write(header_bytes)

create_empty_safetensors('model.safetensors')
create_empty_safetensors('input.safetensors')

print("Test files created successfully")
