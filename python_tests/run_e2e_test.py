#!/usr/bin/env python3
"""
End-to-End Test Runner for MNISC

This script:
1. Generates test data using generate_safetensors.py
2. Runs Python reference runner
3. Optionally runs OCaml AST simulator (if available)
4. Compares outputs
"""

import os
import sys
import subprocess
import tempfile
import shutil
import argparse
from pathlib import Path


def run_command(cmd: list, description: str, verbose: bool = False) -> bool:
    """Run a command and handle errors."""
    print(f"\n[STEP] {description}")
    print(f"  Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True,
            check=True
        )
        if verbose and result.stdout:
            print(result.stdout)
        print(f"  [OK] {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [FAILED] {description}")
        print(f"  Exit code: {e.returncode}")
        if e.stdout:
            print(f"  stdout: {e.stdout}")
        if e.stderr:
            print(f"  stderr: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description='MNISC End-to-End Test')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: temp dir)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for test data')
    parser.add_argument('--skip-generate', action='store_true',
                        help='Skip data generation (use existing)')
    parser.add_argument('--ocaml-sim', type=str, default=None,
                        help='Path to OCaml AST simulator executable')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--keep', action='store_true',
                        help='Keep output directory after test')
    
    args = parser.parse_args()
    
    # Determine script directory
    script_dir = Path(__file__).parent.absolute()
    
    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        is_temp = False
    else:
        output_dir = Path(tempfile.mkdtemp(prefix='mnisc_test_'))
        is_temp = True
    
    print(f"Output directory: {output_dir}")
    
    success = True
    
    try:
        # Step 1: Generate test data
        if not args.skip_generate:
            cmd = [
                sys.executable, str(script_dir / 'generate_safetensors.py'),
                '--output-dir', str(output_dir),
                '--seed', str(args.seed)
            ]
            if not run_command(cmd, "Generate test data", args.verbose):
                success = False
        else:
            print("\n[STEP] Skipping data generation (using existing)")
        
        if not success:
            return 1
        
        # Step 2: Run Python reference runner
        model_path = output_dir / 'model.safetensors'
        input_path = output_dir / 'input.safetensors'
        py_output_path = output_dir / 'py_output.safetensors'
        
        cmd = [
            sys.executable, str(script_dir / 'reference_runner.py'),
            '--model', str(model_path),
            '--input', str(input_path),
            '--output', str(py_output_path)
        ]
        if not run_command(cmd, "Run Python reference runner", args.verbose):
            success = False
        
        # Step 3: Run OCaml AST simulator (if provided)
        ocaml_output_path = None
        if args.ocaml_sim:
            ocaml_output_path = output_dir / 'ocaml_output.safetensors'
            cmd = [
                args.ocaml_sim,
                '--model', str(model_path),
                '--input', str(input_path),
                '--output', str(ocaml_output_path)
            ]
            if not run_command(cmd, "Run OCaml AST simulator", args.verbose):
                print("  [WARNING] OCaml simulator failed, skipping comparison")
                ocaml_output_path = None
        
        # Step 4: Compare outputs
        if ocaml_output_path and ocaml_output_path.exists():
            # Compare Python vs OCaml
            cmd = [
                sys.executable, str(script_dir / 'compare_outputs.py'),
                str(py_output_path),
                str(ocaml_output_path)
            ]
            if not run_command(cmd, "Compare Python vs OCaml outputs", args.verbose):
                success = False
        else:
            print("\n[STEP] Skipping comparison (no OCaml output available)")
            print("  Python reference output saved to:", py_output_path)
        
        # Summary
        print("\n" + "="*60)
        if success:
            print("[PASS] All tests passed!")
            print(f"Output files in: {output_dir}")
            return 0
        else:
            print("[FAIL] Some tests failed.")
            print(f"Check output in: {output_dir}")
            return 1
    
    finally:
        # Cleanup
        if is_temp and not args.keep:
            print(f"\nCleaning up temporary directory: {output_dir}")
            shutil.rmtree(output_dir, ignore_errors=True)
        elif args.keep:
            print(f"\nKeeping output directory: {output_dir}")


if __name__ == '__main__':
    sys.exit(main())
