# MNISC Compiler Makefile

.PHONY: all build clean test

all: build

build:
	cd compiler && dune build

clean:
	cd compiler && dune clean

test: build
	cd compiler && dune exec ./main.exe -- --gen-weights --run-sim

install:
	cd compiler && dune install

fmt:
	cd compiler && dune fmt
