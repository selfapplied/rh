#!/usr/bin/env python3
"""
Test file with a broken import to test the fixed import fixer.
"""

# This import should be broken - wrong module name but function exists
from code.riemann.wrong_module import CE1Visualizer

def test_function():
    visualizer = CE1Visualizer()
    return visualizer
