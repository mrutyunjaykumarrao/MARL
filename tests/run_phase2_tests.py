"""
Phase 2 Test Runner
===================

Runs all Phase 2 tests (DBSCAN clustering + Enemy swarm dynamics).

Usage:
    python tests/run_phase2_tests.py

Author: MARL Jammer Team
"""

import subprocess
import sys
import os


def run_tests():
    """Run all Phase 2 tests."""
    
    # Change to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    print("=" * 70)
    print("PHASE 2 TEST SUITE")
    print("DBSCAN Clustering + Enemy Swarm Dynamics")
    print("=" * 70)
    
    test_files = [
        "tests/test_dbscan.py",
        "tests/test_enemy_swarm.py",
    ]
    
    all_passed = True
    results = []
    
    for test_file in test_files:
        print(f"\n{'=' * 70}")
        print(f"Running: {test_file}")
        print("=" * 70)
        
        result = subprocess.run(
            [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"],
            cwd=project_root
        )
        
        passed = result.returncode == 0
        results.append((test_file, passed))
        
        if not passed:
            all_passed = False
    
    # Summary
    print("\n" + "=" * 70)
    print("PHASE 2 TEST SUMMARY")
    print("=" * 70)
    
    for test_file, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {test_file}: {status}")
    
    print("=" * 70)
    
    if all_passed:
        print("ALL PHASE 2 TESTS PASSED!")
        print("=" * 70)
        return 0
    else:
        print("SOME TESTS FAILED!")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
