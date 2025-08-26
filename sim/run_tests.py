#!/usr/bin/env python3
# Purpose: SIM test runner with comprehensive reporting
# Author: WicketWise AI, Last Modified: 2024

"""
WicketWise SIM Test Runner

Runs comprehensive test suite for the Simulator & Market Replay system.
Provides detailed reporting and coverage analysis.

Usage:
    python sim/run_tests.py [options]
    
Options:
    --verbose, -v     Verbose output
    --coverage, -c    Run with coverage analysis
    --fast, -f        Run only fast tests
    --integration     Run only integration tests
    --unit           Run only unit tests
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path
import time
from datetime import datetime

# Add sim directory to path
sim_dir = Path(__file__).parent
sys.path.insert(0, str(sim_dir))

def run_command(cmd, capture_output=True):
    """Run shell command and return result"""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=capture_output, 
            text=True,
            cwd=sim_dir
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = ['pytest', 'pytest-cov', 'pytest-html']
    missing_packages = []
    
    for package in required_packages:
        success, _, _ = run_command(f"python -c 'import {package.replace('-', '_')}'")
        if not success:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        
        for package in missing_packages:
            success, stdout, stderr = run_command(f"pip install {package}")
            if success:
                print(f"✅ Installed {package}")
            else:
                print(f"❌ Failed to install {package}: {stderr}")
                return False
    
    print("✅ All dependencies available")
    return True

def run_unit_tests(verbose=False, coverage=False):
    """Run unit tests"""
    print("\n🧪 Running Unit Tests...")
    
    test_files = [
        "tests/test_matching.py",
        "tests/test_replay_adapter.py", 
        "tests/test_strategy_baselines.py",
        "tests/test_dgl_enforcement.py",
        "tests/test_metrics.py"
    ]
    
    cmd_parts = ["python", "-m", "pytest"]
    
    if verbose:
        cmd_parts.append("-v")
    
    if coverage:
        cmd_parts.extend(["--cov=sim", "--cov-report=html", "--cov-report=term"])
    
    cmd_parts.extend(test_files)
    cmd = " ".join(cmd_parts)
    
    start_time = time.time()
    success, stdout, stderr = run_command(cmd, capture_output=False)
    duration = time.time() - start_time
    
    if success:
        print(f"✅ Unit tests passed in {duration:.1f}s")
    else:
        print(f"❌ Unit tests failed after {duration:.1f}s")
        if stderr:
            print(f"Error: {stderr}")
    
    return success

def run_integration_tests(verbose=False):
    """Run integration tests"""
    print("\n🔗 Running Integration Tests...")
    
    cmd_parts = ["python", "-m", "pytest"]
    
    if verbose:
        cmd_parts.append("-v")
    
    cmd_parts.append("tests/test_sim_integration.py")
    cmd = " ".join(cmd_parts)
    
    start_time = time.time()
    success, stdout, stderr = run_command(cmd, capture_output=False)
    duration = time.time() - start_time
    
    if success:
        print(f"✅ Integration tests passed in {duration:.1f}s")
    else:
        print(f"❌ Integration tests failed after {duration:.1f}s")
        if stderr:
            print(f"Error: {stderr}")
    
    return success

def run_fast_tests(verbose=False):
    """Run only fast tests (excluding slow integration tests)"""
    print("\n⚡ Running Fast Tests...")
    
    cmd_parts = ["python", "-m", "pytest", "-m", "not slow"]
    
    if verbose:
        cmd_parts.append("-v")
    
    cmd_parts.append("tests/")
    cmd = " ".join(cmd_parts)
    
    start_time = time.time()
    success, stdout, stderr = run_command(cmd, capture_output=False)
    duration = time.time() - start_time
    
    if success:
        print(f"✅ Fast tests passed in {duration:.1f}s")
    else:
        print(f"❌ Fast tests failed after {duration:.1f}s")
    
    return success

def run_all_tests(verbose=False, coverage=False):
    """Run all tests"""
    print("\n🚀 Running All Tests...")
    
    cmd_parts = ["python", "-m", "pytest"]
    
    if verbose:
        cmd_parts.append("-v")
    
    if coverage:
        cmd_parts.extend([
            "--cov=sim", 
            "--cov-report=html:htmlcov",
            "--cov-report=term-missing",
            "--cov-report=xml"
        ])
    
    # Generate HTML report
    cmd_parts.extend([
        "--html=test_report.html",
        "--self-contained-html"
    ])
    
    cmd_parts.append("tests/")
    cmd = " ".join(cmd_parts)
    
    start_time = time.time()
    success, stdout, stderr = run_command(cmd, capture_output=False)
    duration = time.time() - start_time
    
    if success:
        print(f"✅ All tests passed in {duration:.1f}s")
    else:
        print(f"❌ Some tests failed after {duration:.1f}s")
    
    return success

def generate_test_report():
    """Generate comprehensive test report"""
    print("\n📊 Generating Test Report...")
    
    report_content = f"""
# WicketWise SIM Test Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Test Coverage

- **Unit Tests:** Core component functionality
- **Integration Tests:** End-to-end simulation flows  
- **Performance Tests:** Benchmarks and memory usage
- **Configuration Tests:** Validation and serialization

## Test Categories

### 🔧 Matching Engine Tests
- Price-time priority matching
- Partial fills and slippage
- Latency simulation
- Commission calculation
- Market suspension handling

### 📊 Strategy Tests  
- EdgeKelly implementation
- MeanRevert LOB strategy
- Momentum following
- Strategy factory
- Parameter validation

### 🛡️ DGL Enforcement Tests
- Bankroll limit enforcement
- Market exposure caps
- P&L protection rules
- Audit logging
- Violation tracking

### 📈 Metrics Tests
- KPI calculations (Sharpe, Sortino, Drawdown)
- Trade tracking
- Performance snapshots
- Time series export

### 🔄 Adapter Tests
- Replay synchronization
- Mock data generation
- Event processing
- Market state updates

### 🎯 Integration Tests
- End-to-end simulation flows
- Multi-strategy comparison
- Reproducibility validation
- Artifact generation
- Error handling

## Files Tested

"""
    
    # List all test files
    test_files = list(Path("tests").glob("test_*.py"))
    for test_file in test_files:
        report_content += f"- `{test_file}`\n"
    
    report_content += f"""

## Coverage Reports

- **HTML Report:** `htmlcov/index.html`
- **XML Report:** `coverage.xml`
- **Test Report:** `test_report.html`

## Running Tests

```bash
# All tests with coverage
python sim/run_tests.py --coverage

# Fast tests only
python sim/run_tests.py --fast

# Unit tests only  
python sim/run_tests.py --unit

# Integration tests only
python sim/run_tests.py --integration

# Verbose output
python sim/run_tests.py --verbose
```

## Test Results Summary

See `test_report.html` for detailed results including:
- Pass/fail status for each test
- Execution times
- Error details
- Coverage metrics

---
*Generated by WicketWise SIM Test Runner*
"""
    
    with open("TEST_REPORT.md", "w") as f:
        f.write(report_content)
    
    print("✅ Test report generated: TEST_REPORT.md")

def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description="WicketWise SIM Test Runner")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--coverage", "-c", action="store_true", help="Run with coverage")
    parser.add_argument("--fast", "-f", action="store_true", help="Run only fast tests")
    parser.add_argument("--integration", action="store_true", help="Run only integration tests")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    
    args = parser.parse_args()
    
    print("🏏 WicketWise SIM Test Runner")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    success = True
    
    try:
        if args.unit:
            success = run_unit_tests(args.verbose, args.coverage)
        elif args.integration:
            success = run_integration_tests(args.verbose)
        elif args.fast:
            success = run_fast_tests(args.verbose)
        else:
            success = run_all_tests(args.verbose, args.coverage)
        
        # Generate report
        generate_test_report()
        
        if success:
            print("\n🎉 All tests completed successfully!")
            
            if args.coverage:
                print("\n📊 Coverage reports generated:")
                print("  - HTML: htmlcov/index.html")
                print("  - XML: coverage.xml")
            
            print("  - Test Report: test_report.html")
            print("  - Summary: TEST_REPORT.md")
            
        else:
            print("\n❌ Some tests failed. Check the reports for details.")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test runner error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
