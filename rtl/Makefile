#------------------------------------------------------------------------------
# MNISC EU Verilator Makefile
# Builds Verilator harness for EU RTL simulation
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------
VERILATOR ?= verilator

# Directories
RTL_DIR := .
OBJ_DIR := obj_dir

# C++ Compiler
CXX ?= g++
CXXFLAGS ?= -std=c++17 -Wall -O2

# Verilator generated files
VM_PREFIX := Veu_top
VM_DIR := $(OBJ_DIR)
VEXE := $(VM_DIR)/$(VM_PREFIX)

#------------------------------------------------------------------------------
# RTL Source Files (SystemVerilog) - order matters: packages first, then modules
#------------------------------------------------------------------------------
RTL_SRCS := $(RTL_DIR)/eu_isa_pkg.sv \
            $(RTL_DIR)/muladd2_lut.sv \
            $(RTL_DIR)/feature_line_buffer.sv \
            $(RTL_DIR)/weight_buffer.sv \
            $(RTL_DIR)/conv_core_lowbit.sv \
            $(RTL_DIR)/gemm_core_lowbit.sv \
            $(RTL_DIR)/pool2d_unit.sv \
            $(RTL_DIR)/unpool2d_unit.sv \
            $(RTL_DIR)/concat_unit.sv \
            $(RTL_DIR)/act_quant_unit.sv \
            $(RTL_DIR)/eu_sequencer.sv \
            $(RTL_DIR)/eu_top.sv

#------------------------------------------------------------------------------
# Testbench Source
#------------------------------------------------------------------------------
TB_SRC := $(RTL_DIR)/tb_eu_top.cpp

#------------------------------------------------------------------------------
# Include paths for C++
#------------------------------------------------------------------------------
INCLUDES := -I$(VM_DIR)

#------------------------------------------------------------------------------
# Verilator Compilation Flags
#------------------------------------------------------------------------------
VERILATOR_FLAGS := --cc --exe --build \
                   -Mdir $(VM_DIR) \
                   --top-module eu_top \
                   -CFLAGS "$(CXXFLAGS) $(INCLUDES)" \
                   --trace \
                   --trace-structs \
                   --assert \
                   --x-assign unique \
                   --x-initial unique

#------------------------------------------------------------------------------
# Build Targets
#------------------------------------------------------------------------------

.PHONY: all clean run run_trace run_bp wave verilator_version help

# Default target: build the executable
all: $(VEXE)

# Build the Verilator executable
$(VEXE): $(RTL_SRCS) $(TB_SRC)
	@echo "================================================"
	@echo "Building MNISC EU Verilator Model"
	@echo "================================================"
	@mkdir -p $(VM_DIR)
	$(VERILATOR) $(VERILATOR_FLAGS) $(RTL_SRCS) $(TB_SRC)
	@echo "Build complete: $@"

# Run simulation
run: $(VEXE)
	@echo "================================================"
	@echo "Running MNISC EU Simulation"
	@echo "================================================"
	cd $(VM_DIR) && ./$(VM_PREFIX) +program=../program.bin +meta=../program_meta.json

# Run with trace enabled
run_trace: $(VEXE)
	@echo "================================================"
	@echo "Running MNISC EU Simulation (with VCD trace)"
	@echo "================================================"
	cd $(VM_DIR) && ./$(VM_PREFIX) +program=../program.bin +meta=../program_meta.json +trace

# Run with backpressure testing
run_bp: $(VEXE)
	@echo "================================================"
	@echo "Running MNISC EU Simulation (with backpressure)"
	@echo "================================================"
	cd $(VM_DIR) && ./$(VM_PREFIX) +program=../program.bin +meta=../program_meta.json +backpressure

# View waveforms
wave: $(VM_DIR)/eu_top.vcd
	@if command -v gtkwave >/dev/null 2>&1; then \
		gtkwave $< & \
	else \
		echo "gtkwave not found. Install with: apt-get install gtkwave"; \
	fi

# Check Verilator version
verilator_version:
	$(VERILATOR) --version

#------------------------------------------------------------------------------
# Clean
#------------------------------------------------------------------------------

clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(VM_DIR)
	rm -f *.vcd
	rm -f verilator_output.safetensors
	rm -f *.log

distclean: clean
	rm -f program.bin program_meta.json
	rm -f model.safetensors input.safetensors

#------------------------------------------------------------------------------
# Debug Targets
#------------------------------------------------------------------------------

xml: $(RTL_SRCS)
	$(VERILATOR) --xml-only -Mdir $(VM_DIR) --top-module eu_top $(RTL_SRCS)

preproc: $(RTL_SRCS)
	$(VERILATOR) -E -Mdir $(VM_DIR) --top-module eu_top $(RTL_SRCS)

#------------------------------------------------------------------------------
# Help
#------------------------------------------------------------------------------

help:
	@echo "MNISC EU Verilator Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all           - Build the Verilator executable (default)"
	@echo "  run           - Run simulation with default inputs"
	@echo "  run_trace     - Run simulation with VCD trace enabled"
	@echo "  run_bp        - Run simulation with backpressure testing"
	@echo "  wave          - Open VCD waveform in gtkwave"
	@echo "  clean         - Remove build artifacts"
	@echo "  distclean     - Remove all generated files"
	@echo "  verilator_version - Show Verilator version"
	@echo ""
	@echo "Input files (expected in current directory):"
	@echo "  program.bin         - u32 instruction stream"
	@echo "  program_meta.json   - Instruction metadata"
	@echo "  model.safetensors   - Model weights"
	@echo "  input.safetensors   - Input activations"
	@echo ""
	@echo "Output files:"
	@echo "  verilator_output.safetensors - Simulation output"
	@echo "  eu_top.vcd                   - VCD waveform (if +trace)"
