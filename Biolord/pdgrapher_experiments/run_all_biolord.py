#!/usr/bin/env python3
"""
Wrapper script to run biolord training for all PDGrapher cell types.

Usage:
    # Run all cell types, all folds (sequentially)
    python run_all_biolord.py --gpu 6
    
    # Run specific cell types
    python run_all_biolord.py --gpu 6 --cells A375 MCF7 PC3
    
    # Run specific folds
    python run_all_biolord.py --gpu 6 --folds 1 2 3
    python train_bl_pdg_org.py --test_cell A375 --fold 1 --gpu 6 --max_epochs 10
    # Run with more epochs
    python run_all_biolord.py --gpu 6 --max_epochs 500
    
    
    python run_all_biolord.py --gpu 6 --script train_bl_pdg_org.py --cells A375 --folds 1 --max_epochs 10
    
    python run_all_biolord.py --gpu 1 --cells MCF7 --max_epoch 100
"""

import subprocess
import argparse
import sys
from datetime import datetime
from pathlib import Path

# All cell types in PDGrapher
ALL_CELL_TYPES = [
    'A375',
    'A549', 
    'HT29',
    'MCF7',
    'PC3',
    'VCAP',
    'HELA',
    'MDAMB231',
    'BT20',
]

ALL_FOLDS = [1, 2, 3, 4, 5]


def run_training(cell_type, fold, gpu, max_epochs, output_dir, script_path, extra_args):
    """Run training for a single cell type and fold."""
    cmd = [
        sys.executable,
        str(script_path),
        '--test_cell', cell_type,
        '--fold', str(fold),
        '--gpu', str(gpu),
        '--max_epochs', str(max_epochs),
        '--output_dir', str(output_dir),
    ]
    cmd.extend(extra_args)
    
    print(f"\n{'='*70}")
    print(f"RUNNING: {cell_type} fold {fold}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}\n", flush=True)
    
    start_time = datetime.now()
    
    # Run subprocess and stream output to terminal in real-time
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    
    # Stream output line by line
    for line in process.stdout:
        print(line, end='', flush=True)
    
    process.wait()
    returncode = process.returncode
    
    end_time = datetime.now()
    runtime = end_time - start_time
    
    status = "SUCCESS" if returncode == 0 else "FAILED"
    print(f"\n{status}: {cell_type} fold {fold} (runtime: {runtime})", flush=True)
    
    return returncode == 0, runtime


def main():
    parser = argparse.ArgumentParser(description='Run biolord training for all PDGrapher cell types')
    
    parser.add_argument('--gpu', type=str, required=True,
                        help='GPU device ID')
    parser.add_argument('--cells', type=str, nargs='+', default=None,
                        help=f'Cell types to run (default: all). Options: {ALL_CELL_TYPES}')
    parser.add_argument('--folds', type=int, nargs='+', default=None,
                        help='Folds to run (default: 1-5)')
    parser.add_argument('--max_epochs', type=int, default=500,
                        help='Max training epochs (default: 500, official biolord setting)')
    parser.add_argument('--output_dir', type=str,
                        default='/raid/home/joshua/projects/biolord/pdgrapher_experiments/output',
                        help='Output directory')
    parser.add_argument('--script', type=str, default='train_bl_pdg_org.py',
                        help='Path to training script')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print commands without executing')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable wandb logging')
    
    args, extra_args = parser.parse_known_args()
    
    # Determine cell types and folds to run
    cell_types = args.cells if args.cells else ALL_CELL_TYPES
    folds = args.folds if args.folds else ALL_FOLDS
    
    # Validate cell types
    for cell in cell_types:
        if cell not in ALL_CELL_TYPES:
            print(f"WARNING: {cell} not in known cell types: {ALL_CELL_TYPES}")
    
    # Add no_wandb flag if requested
    if args.no_wandb:
        extra_args.append('--no_wandb')
    
    # Find script
    script_path = Path(args.script)
    if not script_path.exists():
        # Try in same directory as this script
        script_path = Path(__file__).parent / args.script
    if not script_path.exists():
        # Try absolute path
        script_path = Path(args.script).resolve()
    if not script_path.exists():
        print(f"ERROR: Training script not found: {args.script}")
        print(f"  Tried:")
        print(f"    - {args.script}")
        print(f"    - {Path(__file__).parent / args.script}")
        print(f"  Please provide the correct path with --script")
        sys.exit(1)
    
    script_path = script_path.resolve()
    print(f"Script found: {script_path}")
    
    # Summary
    total_runs = len(cell_types) * len(folds)
    print("="*70)
    print("BIOLORD TRAINING - ALL CELL TYPES")
    print("="*70)
    print(f"Cell types: {cell_types}")
    print(f"Folds: {folds}")
    print(f"Total runs: {total_runs}")
    print(f"GPU: {args.gpu}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Output dir: {args.output_dir}")
    print(f"Script: {script_path}")
    print("="*70)
    
    if args.dry_run:
        print("\n[DRY RUN - Commands that would be executed:]\n")
        for cell in cell_types:
            for fold in folds:
                cmd = [
                    sys.executable, str(script_path),
                    '--test_cell', cell,
                    '--fold', str(fold),
                    '--gpu', str(args.gpu),
                    '--max_epochs', str(args.max_epochs),
                    '--output_dir', str(args.output_dir),
                ] + extra_args
                print(f"  {' '.join(cmd)}")
        print("\n[End dry run]")
        return
    
    # Run all combinations
    results = []
    total_start = datetime.now()
    
    for cell in cell_types:
        for fold in folds:
            success, runtime = run_training(
                cell_type=cell,
                fold=fold,
                gpu=args.gpu,
                max_epochs=args.max_epochs,
                output_dir=args.output_dir,
                script_path=script_path,
                extra_args=extra_args,
            )
            results.append({
                'cell': cell,
                'fold': fold,
                'success': success,
                'runtime': runtime,
            })
    
    total_end = datetime.now()
    total_runtime = total_end - total_start
    
    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"{'Cell Type':<12} {'Fold':<6} {'Status':<10} {'Runtime':<15}")
    print("-"*70)
    
    n_success = 0
    n_failed = 0
    
    for r in results:
        status = "SUCCESS" if r['success'] else "FAILED"
        if r['success']:
            n_success += 1
        else:
            n_failed += 1
        print(f"{r['cell']:<12} {r['fold']:<6} {status:<10} {str(r['runtime']):<15}")
    
    print("-"*70)
    print(f"Total: {n_success} succeeded, {n_failed} failed")
    print(f"Total runtime: {total_runtime}")
    print("="*70)
    
    # Exit with error if any failed
    if n_failed > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()