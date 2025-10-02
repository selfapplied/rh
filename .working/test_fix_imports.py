#!/usr/bin/env python3
"""
Test scaffolding for the import fixer tool.

This creates test scenarios to validate the import fixer before
it can be promoted to stable tooling.
"""

import os
import tempfile
import shutil
from pathlib import Path
import subprocess
import sys


def create_test_repository():
    """Create a test repository with broken imports."""
    test_dir = Path(tempfile.mkdtemp(prefix="import_fix_test_"))
    
    # Initialize git repo
    subprocess.run(["git", "init"], cwd=test_dir, check=True)
    
    # Create test structure
    (test_dir / "code" / "math").mkdir(parents=True)
    (test_dir / "code" / "utils").mkdir(parents=True)
    (test_dir / "tools" / "helpers").mkdir(parents=True)
    
    # Create test files with broken imports
    test_files = {
        "code/math/calculator.py": '''
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b
''',
        "code/utils/formatter.py": '''
def format_number(num):
    return f"{num:,}"

def format_currency(amount):
    return f"${amount:.2f}"
''',
        "tools/helpers/validator.py": '''
def is_positive(num):
    return num > 0

def is_even(num):
    return num % 2 == 0
''',
        "code/math/analysis.py": '''
# This file has broken imports that need fixing
from code.utils.formatter import format_number, format_currency
from tools.helpers.validator import is_positive, is_even
from code.math.calculator import add, multiply

def analyze_data(data):
    """Analyze data using various utilities."""
    if is_positive(data):
        formatted = format_number(data)
        doubled = multiply(data, 2)
        return f"Positive: {formatted}, doubled: {doubled}"
    return "Not positive"
''',
        "code/utils/report.py": '''
# Another file with broken imports
from code.math.calculator import add
from tools.helpers.validator import is_even

def generate_report(numbers):
    """Generate a report of numbers."""
    total = 0
    for num in numbers:
        if is_even(num):
            total = add(total, num)
    return total
'''
    }
    
    # Write test files
    for file_path, content in test_files.items():
        full_path = test_dir / file_path
        full_path.write_text(content)
    
    # Commit initial state
    subprocess.run(["git", "add", "."], cwd=test_dir, check=True)
    subprocess.run(["git", "commit", "-m", "Initial test files"], cwd=test_dir, check=True)
    
    return test_dir


def test_import_fixer():
    """Test the import fixer on the test repository."""
    test_dir = create_test_repository()
    
    try:
        print(f"Created test repository: {test_dir}")
        
        # Copy the fix_imports.py tool to test directory
        tool_path = test_dir / "fix_imports.py"
        shutil.copy(".working/fix_imports.py", tool_path)
        
        # Run the import fixer
        print("\n" + "="*50)
        print("TESTING IMPORT FIXER")
        print("="*50)
        
        result = subprocess.run([
            sys.executable, "fix_imports.py", "code", "tools"
        ], cwd=test_dir, capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        print("\nSTDERR:")
        print(result.stderr)
        print(f"\nReturn code: {result.returncode}")
        
        # Check if files were modified
        print("\n" + "="*50)
        print("CHECKING RESULTS")
        print("="*50)
        
        # Show git diff
        diff_result = subprocess.run([
            "git", "diff"
        ], cwd=test_dir, capture_output=True, text=True)
        
        if diff_result.stdout:
            print("Changes made:")
            print(diff_result.stdout)
        else:
            print("No changes detected")
        
        # Show final file contents
        print("\nFinal file contents:")
        for file_path in ["code/math/analysis.py", "code/utils/report.py"]:
            full_path = test_dir / file_path
            if full_path.exists():
                print(f"\n{file_path}:")
                print("-" * 30)
                print(full_path.read_text())
        
        return result.returncode == 0
        
    finally:
        # Clean up
        shutil.rmtree(test_dir)
        print(f"\nCleaned up test directory: {test_dir}")


def test_safety_checks():
    """Test that safety checks work properly."""
    print("\n" + "="*50)
    print("TESTING SAFETY CHECKS")
    print("="*50)
    
    # Test 1: Not in git repository
    with tempfile.TemporaryDirectory() as temp_dir:
        print("\nTest 1: Not in git repository")
        result = subprocess.run([
            sys.executable, ".working/fix_imports.py", "code"
        ], cwd=temp_dir, capture_output=True, text=True)
        print(f"Return code: {result.returncode}")
        print("Output:", result.stdout)
    
    # Test 2: Dirty working directory
    test_dir = create_test_repository()
    try:
        print("\nTest 2: Dirty working directory")
        # Make a change
        (test_dir / "code" / "math" / "calculator.py").write_text("def add(a, b):\n    return a + b + 1  # Modified")
        
        result = subprocess.run([
            sys.executable, "fix_imports.py", "code"
        ], cwd=test_dir, capture_output=True, text=True)
        print(f"Return code: {result.returncode}")
        print("Output:", result.stdout)
        
    finally:
        shutil.rmtree(test_dir)


if __name__ == "__main__":
    print("Import Fixer Test Suite")
    print("=" * 50)
    
    # Test safety checks first
    test_safety_checks()
    
    # Test the actual fixer
    success = test_import_fixer()
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
