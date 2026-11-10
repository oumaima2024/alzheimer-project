"""
run_all.py
----------
Convenience script to run the complete RQ3 pipeline in sequence.

Usage:
  python run_all.py
  python run_all.py --config config.yaml --data-dir data/raw
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def run(cmd, cwd=None):
    print(f'\n{"="*60}')
    print(f'Running: {" ".join(cmd)}')
    print('='*60)
    result = subprocess.run(cmd, cwd=cwd or str(Path(__file__).parent))
    if result.returncode != 0:
        print(f'ERROR: Command failed with code {result.returncode}')
        sys.exit(result.returncode)


def main():
    p = argparse.ArgumentParser(description='Run full RQ3 pipeline')
    p.add_argument('--config', default='config.yaml')
    p.add_argument('--data-dir', default=None)
    p.add_argument('--results-dir', default='results')
    p.add_argument('--model-type', default='GRU', choices=['GRU', 'LSTM'])
    p.add_argument('--skip-tests', action='store_true')
    args = p.parse_args()

    base_dir = str(Path(__file__).parent)

    # 1. Tests
    if not args.skip_tests:
        run([sys.executable, '-m', 'pytest', 'tests/', '-v', '--tb=short'],
            cwd=base_dir)

    # 2. Baseline training
    cmd = [sys.executable, 'scripts/train_baseline.py',
           '--config', args.config]
    if args.data_dir:
        cmd += ['--data-dir', args.data_dir]
    if args.results_dir:
        cmd += ['--results-dir', args.results_dir]
    run(cmd, cwd=base_dir)

    # 3. Longitudinal training
    cmd = [sys.executable, 'scripts/train_longitudinal.py',
           '--config', args.config,
           '--model-type', args.model_type]
    if args.data_dir:
        cmd += ['--data-dir', args.data_dir]
    if args.results_dir:
        cmd += ['--results-dir', args.results_dir]
    run(cmd, cwd=base_dir)

    # 4. Compare
    run([sys.executable, 'scripts/compare_models.py',
         '--results-dir', args.results_dir],
        cwd=base_dir)

    print('\n' + '='*60)
    print('Pipeline complete. Check results/ for outputs.')
    print('='*60)


if __name__ == '__main__':
    main()
