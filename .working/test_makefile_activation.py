#!/usr/bin/env python3
"""Test if we can modify Makefile to activate environment once."""
import subprocess
import os

def test_activation_approach():
    """Test running commands after activating the environment."""
    
    # Test 1: Activate and run in same command
    print("Test 1: Activate and run in same command")
    result = subprocess.run([
        'bash', '-c', 
        'source .venv/bin/activate && python -m tools.check --help'
    ], capture_output=True, text=True)
    
    print(f"Exit code: {result.returncode}")
    if result.returncode == 0:
        print("✅ Single command activation works!")
    else:
        print(f"❌ Failed: {result.stderr}")
    
    # Test 2: Check if we can set up environment variables
    print("\nTest 2: Using environment variables")
    env = os.environ.copy()
    env['PYTHONPATH'] = '.'
    
    # Get the poetry python path
    result = subprocess.run(['poetry', 'run', 'python', '-c', 'import sys; print(sys.executable)'], 
                          capture_output=True, text=True, env=env)
    if result.returncode == 0:
        poetry_python = result.stdout.strip()
        print(f"Poetry python: {poetry_python}")
        
        # Test running directly with that python
        result2 = subprocess.run([poetry_python, '-m', 'tools.check', '--help'], 
                               capture_output=True, text=True, env=env)
        print(f"Direct python result: {result2.returncode}")
        if result2.returncode == 0:
            print("✅ Direct python path works!")

if __name__ == "__main__":
    test_activation_approach()
