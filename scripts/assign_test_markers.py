#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Automated Test Marker Assignment Script

This script assigns pytest markers (@pytest.mark.single_rank or @pytest.mark.multi_rank_required)
to test files based on the type of functionality they test.

Classification rules:
- single_rank: Tests validating tensor properties (shape, dtype, values) on symmetric heap
  Examples: zeros, ones, empty, full, rand, randint, randn, arange, linspace
  
- multi_rank_required: Tests validating distributed behavior and cross-rank operations
  Examples: get, put, load, store, atomic operations, broadcast, copy, all_reduce, all_gather, all_to_all
"""

import os
import sys
import re
from pathlib import Path


# Tests that should be marked as single_rank (tensor property tests)
SINGLE_RANK_PATTERNS = [
    "test_zeros.py",
    "test_ones.py", 
    "test_empty.py",
    "test_full.py",
    "test_rand.py",
    "test_randint.py",
    "test_randn.py",
    "test_arange.py",
    "test_linspace.py",
    "test_zeros_like.py",
]

# Tests that should be marked as multi_rank_required (distributed tests)
MULTI_RANK_PATTERNS = [
    # Remote memory access operations
    "test_get_gluon.py",
    "test_get_triton.py",
    "test_put_gluon.py",
    "test_put_triton.py",
    "test_load_gluon.py",
    "test_load_triton.py",
    "test_store_gluon.py",
    "test_store_triton.py",
    # Atomic operations
    "test_atomic_add_gluon.py",
    "test_atomic_add_triton.py",
    "test_atomic_and_gluon.py",
    "test_atomic_and_triton.py",
    "test_atomic_cas_gluon.py",
    "test_atomic_cas_triton.py",
    "test_atomic_max_gluon.py",
    "test_atomic_max_triton.py",
    "test_atomic_min_gluon.py",
    "test_atomic_min_triton.py",
    "test_atomic_or_gluon.py",
    "test_atomic_or_triton.py",
    "test_atomic_xchg_gluon.py",
    "test_atomic_xchg_triton.py",
    "test_atomic_xor_gluon.py",
    "test_atomic_xor_triton.py",
    # Data movement operations
    "test_broadcast_gluon.py",
    "test_broadcast_triton.py",
    "test_copy_gluon.py",
    "test_copy_triton.py",
    # Collective operations (all in ccl, ops, x directories)
    "test_all_reduce.py",
    "test_all_gather.py",
    "test_all_to_all.py",
    "test_all_to_all_gluon.py",
    "test_process_groups.py",
    "test_reduce_scatter.py",
    "test_gather.py",
    # Matmul + collective operations
    "test_all_gather_matmul.py",
    "test_matmul_all_gather.py",
    "test_matmul_all_reduce.py",
    "test_matmul_reduce_scatter.py",
]

# Tests in examples directory that test distributed behavior
EXAMPLE_MULTI_RANK_PATTERNS = [
    "test_load_bench.py",
    "test_all_load_bench.py",
    "test_atomic_add_bench.py",
    "test_message_passing.py",
    "test_flash_decode.py",
]


def should_mark_single_rank(filepath: Path) -> bool:
    """Check if a test file should be marked as single_rank."""
    filename = filepath.name
    return filename in SINGLE_RANK_PATTERNS


def should_mark_multi_rank(filepath: Path) -> bool:
    """Check if a test file should be marked as multi_rank_required."""
    filename = filepath.name
    
    # Check if it's in the patterns list
    if filename in MULTI_RANK_PATTERNS:
        return True
    
    # Check if it's in examples directory and matches example patterns
    if "examples" in filepath.parts and filename in EXAMPLE_MULTI_RANK_PATTERNS:
        return True
    
    return False


def get_marker_for_file(filepath: Path) -> str:
    """Determine the appropriate marker for a test file."""
    if should_mark_single_rank(filepath):
        return "single_rank"
    elif should_mark_multi_rank(filepath):
        return "multi_rank_required"
    else:
        # Leave unmarked for backward compatibility
        return None


def has_marker(content: str, marker: str) -> bool:
    """Check if the file already has the specified marker."""
    marker_pattern = rf"@pytest\.mark\.{marker}"
    return re.search(marker_pattern, content) is not None


def add_marker_to_file(filepath: Path, marker: str, dry_run: bool = False) -> bool:
    """Add a pytest marker to all test functions in a file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Check if marker already exists
    if has_marker(content, marker):
        print(f"  ✓ {filepath.name} already has @pytest.mark.{marker}")
        return False
    
    # Find the first test function or parametrize decorator
    # Add the marker after imports and before the first test/parametrize
    lines = content.split('\n')
    new_lines = []
    marker_added = False
    in_imports = True
    
    for i, line in enumerate(lines):
        new_lines.append(line)
        
        # Check if we're past the imports
        if in_imports and line.strip() and not line.strip().startswith(('#', 'import', 'from', '"""', "'''")):
            in_imports = False
        
        # Add marker before first @pytest.mark.parametrize or def test_
        if not marker_added and not in_imports:
            if line.strip().startswith('@pytest.mark.parametrize') or line.strip().startswith('def test_'):
                # Insert marker before this line
                new_lines.insert(-1, f'\npytestmark = pytest.mark.{marker}\n')
                marker_added = True
                break
    
    if not marker_added:
        # If no test function found, try a different approach
        # Add after the last import
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip().startswith(('import', 'from')):
                lines.insert(i + 1, f'\npytestmark = pytest.mark.{marker}\n')
                marker_added = True
                break
        
        if marker_added:
            new_lines = lines
    
    if not marker_added:
        print(f"  ✗ Could not find appropriate location to add marker in {filepath.name}")
        return False
    
    new_content = '\n'.join(new_lines)
    
    if dry_run:
        print(f"  → Would add @pytest.mark.{marker} to {filepath.name}")
        return True
    else:
        with open(filepath, 'w') as f:
            f.write(new_content)
        print(f"  ✓ Added @pytest.mark.{marker} to {filepath.name}")
        return True


def process_test_directory(test_dir: Path, dry_run: bool = False) -> dict:
    """Process all test files in a directory."""
    stats = {
        'total': 0,
        'single_rank': 0,
        'multi_rank': 0,
        'unmarked': 0,
        'modified': 0,
    }
    
    for test_file in test_dir.rglob('test_*.py'):
        stats['total'] += 1
        marker = get_marker_for_file(test_file)
        
        if marker == 'single_rank':
            stats['single_rank'] += 1
            if add_marker_to_file(test_file, marker, dry_run):
                stats['modified'] += 1
        elif marker == 'multi_rank_required':
            stats['multi_rank'] += 1
            if add_marker_to_file(test_file, marker, dry_run):
                stats['modified'] += 1
        else:
            stats['unmarked'] += 1
            print(f"  - {test_file.name} left unmarked (backward compatibility)")
    
    return stats


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Assign pytest markers to test files based on functionality',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )
    parser.add_argument(
        '--test-dir',
        type=Path,
        default=Path('tests'),
        help='Path to tests directory (default: tests)'
    )
    
    args = parser.parse_args()
    
    if not args.test_dir.exists():
        print(f"Error: Test directory {args.test_dir} does not exist")
        sys.exit(1)
    
    print(f"Processing test files in {args.test_dir}...")
    if args.dry_run:
        print("DRY RUN - no files will be modified\n")
    
    stats = process_test_directory(args.test_dir, args.dry_run)
    
    print("\n" + "="*70)
    print("Summary:")
    print("="*70)
    print(f"Total test files:           {stats['total']}")
    print(f"Single-rank tests:          {stats['single_rank']}")
    print(f"Multi-rank required tests:  {stats['multi_rank']}")
    print(f"Unmarked tests:             {stats['unmarked']}")
    print(f"Files modified:             {stats['modified']}")
    
    if args.dry_run:
        print("\nRun without --dry-run to apply changes")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
