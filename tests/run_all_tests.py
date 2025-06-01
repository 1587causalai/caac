#!/usr/bin/env python3
"""
Run all tests in the tests directory

Usage:
    python run_all_tests.py [--quick]
    
    --quick: Only run quick tests that finish fast
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

def run_test(test_file, verbose=True):
    """运行单个测试文件"""
    print(f"\n{'='*60}")
    print(f"Running {test_file}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=False, 
                              text=True, 
                              cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print(f"✅ {test_file} - PASSED")
            return True
        else:
            print(f"❌ {test_file} - FAILED (exit code: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"❌ {test_file} - ERROR: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run all tests')
    parser.add_argument('--quick', action='store_true', 
                       help='Only run quick tests')
    args = parser.parse_args()
    
    print("🧪 CAAC Project Test Suite")
    print("="*60)
    
    # 定义测试文件和它们的属性
    tests = [
        {
            'file': 'test_new_datasets.py',
            'name': 'Dataset Loading Tests',
            'quick': True,
            'description': 'Test dataset loading functionality'
        },
        {
            'file': 'test_classification_outliers.py', 
            'name': 'Classification Outlier Tests',
            'quick': False,
            'description': 'Test classification data outlier functionality'
        },
        {
            'file': 'test_new_data_split.py',
            'name': 'Data Split Tests', 
            'quick': False,
            'description': 'Test new data splitting strategies'
        },
        {
            'file': 'test_crammer_singer.py',
            'name': 'Crammer-Singer Tests',
            'quick': True, 
            'description': 'Test Crammer-Singer specific functionality'
        }
    ]
    
    # 过滤测试
    if args.quick:
        tests_to_run = [t for t in tests if t['quick']]
        print("🚀 Running QUICK tests only")
    else:
        tests_to_run = tests
        print("🔍 Running ALL tests")
    
    print(f"📋 Tests to run: {len(tests_to_run)}")
    for test in tests_to_run:
        print(f"  - {test['name']}: {test['description']}")
    
    # 运行测试
    results = []
    for test_info in tests_to_run:
        test_file = test_info['file']
        if os.path.exists(test_file):
            success = run_test(test_file)
            results.append((test_file, success))
        else:
            print(f"⚠️  Test file not found: {test_file}")
            results.append((test_file, False))
    
    # 总结结果
    print(f"\n{'='*60}")
    print("📊 TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_file, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {test_file:<35} {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print(f"⚠️  {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 