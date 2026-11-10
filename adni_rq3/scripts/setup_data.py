"""
setup_data.py
-------------
Helper script to copy ADNI CSV files from a source directory
(or parent folder) into data/raw/.

Usage:
  python scripts/setup_data.py --source C:\\path\\to\\adni_csvs
  python scripts/setup_data.py --source ..    # parent folder of adni_rq3
"""

import argparse
import shutil
import os
from pathlib import Path

KNOWN_FILES = {
    'CDR': ['CDR_*.csv', 'CDR*.csv'],
    'ITEM': ['ITEM.csv', 'ITEM*.csv'],
    'DXSUM': ['DXSUM_*.csv', 'DXSUM*.csv'],
    'ADNIMERGE': ['ADNIMERGE.csv'],
}


def find_and_copy(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    found = []
    not_found = []

    for name, patterns in KNOWN_FILES.items():
        matched = []
        for pat in patterns:
            matched.extend(src_dir.glob(pat))
        if matched:
            src = matched[0]
            dst = dst_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
                print(f'  Copied: {src.name} → {dst}')
            else:
                print(f'  Already exists: {dst}')
            found.append(name)
        else:
            not_found.append(name)

    print(f'\nFound: {found}')
    if not_found:
        print(f'Not found: {not_found}')
        print('  → These files are optional or must be downloaded from '
              'adni.loni.usc.edu')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True,
                   help='Directory containing ADNI CSV files')
    p.add_argument('--dest', default='data/raw',
                   help='Destination directory (default: data/raw)')
    args = p.parse_args()

    src = Path(args.source).resolve()
    dst = Path(args.dest).resolve()

    print(f'Scanning: {src}')
    print(f'Destination: {dst}')
    find_and_copy(src, dst)


if __name__ == '__main__':
    main()
