"""
Phase 1 Test Runner
===================

Runs all Phase 1 tests (Physics Foundation) and generates a summary report.

Usage:
    python tests/run_phase1_tests.py

This script tests:
    1. FSPL Module - RF propagation calculations
    2. Communication Graph Module - Adjacency, Laplacian, Lambda-2
    3. Jamming Module - FSPL-based link disruption

Author: MARL Jammer Team
"""

import sys
import os
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))


def run_test_module(module_name: str, run_func) -> dict:
    """
    Run a test module and capture results.
    
    Args:
        module_name: Name of the test module
        run_func: Function that runs the tests
        
    Returns:
        Dict with pass/fail status and timing
    """
    print(f"\n{'#'*70}")
    print(f"# {module_name}")
    print(f"{'#'*70}")
    
    start_time = time.time()
    try:
        success = run_func()
    except Exception as e:
        print(f"\nFATAL ERROR in {module_name}: {type(e).__name__}: {e}")
        success = False
    elapsed = time.time() - start_time
    
    return {
        "module": module_name,
        "success": success,
        "elapsed_seconds": elapsed
    }


def main():
    """Run all Phase 1 tests."""
    print("="*70)
    print(" MARL JAMMER - PHASE 1 TEST SUITE")
    print(" Physics Foundation: FSPL, Communication Graph, Jamming")
    print("="*70)
    
    results = []
    
    # Test 1: FSPL Module
    from test_fspl import run_all_tests as run_fspl_tests
    results.append(run_test_module("FSPL Module", run_fspl_tests))
    
    # Test 2: Communication Graph Module
    from test_laplacian import run_all_tests as run_laplacian_tests
    results.append(run_test_module("Communication Graph Module", run_laplacian_tests))
    
    # Test 3: Jamming Module
    from test_jamming import run_all_tests as run_jamming_tests
    results.append(run_test_module("Jamming Module", run_jamming_tests))
    
    # Summary
    print("\n")
    print("="*70)
    print(" PHASE 1 TEST SUMMARY")
    print("="*70)
    
    total_time = sum(r["elapsed_seconds"] for r in results)
    all_passed = all(r["success"] for r in results)
    
    for r in results:
        status = "PASS" if r["success"] else "FAIL"
        print(f"  [{status}] {r['module']} ({r['elapsed_seconds']:.2f}s)")
    
    print("-"*70)
    print(f"  Total Time: {total_time:.2f}s")
    
    if all_passed:
        print("\n  " + "="*50)
        print("  ALL PHASE 1 TESTS PASSED!")
        print("  " + "="*50)
        print("\n  Phase 1 (Physics Foundation) is complete.")
        print("  You can now proceed to Phase 2 (Environment).")
    else:
        print("\n  " + "="*50)
        print("  SOME TESTS FAILED!")
        print("  " + "="*50)
        print("\n  Please fix failing tests before proceeding.")
    
    print("="*70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
