#!/usr/bin/env python3
"""
Test file with broken imports.
"""

# These imports are broken
from nonexistent_module import some_function
import another_missing_module
from riemann.pascal import pascal_nested_brackets  # This function was renamed

def test_function():
    result = some_function()
    another_missing_module.do_something()
    brackets = pascal_nested_brackets(0.5, 1, 5)
    return result, brackets

if __name__ == "__main__":
    test_function()
