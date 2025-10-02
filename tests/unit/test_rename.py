#!/usr/bin/env python3
"""
Test file for symbol rename detection.
"""

# This import should fail and trigger rename detection
from riemann.pascal import pascal_nested_brackets

def test_function():
    # Use the old function name
    result = pascal_nested_brackets(0.5, 1, 5)
    return result

if __name__ == "__main__":
    test_function()
