# Import Fixer Development

This directory contains experimental tools for automatic import fixing.

## Tools

- `fix_imports.py` - The main import fixing tool (experimental)
- `test_fix_imports.py` - Test scaffolding and validation

## Development Workflow

### 1. Test the Tool
```bash
# Run comprehensive tests
python .working/test_fix_imports.py
```

### 2. Manual Testing
```bash
# Test on a small subset
python .working/fix_imports.py code/riemann/some_file.py

# Test on entire directories
python .working/fix_imports.py code/tools/
```

### 3. Safety Validation
The tool includes safety checks:
- ✅ Must be in git repository
- ✅ Working directory must be clean
- ✅ Validates fixes with syntax checking

### 4. Promotion Criteria

Before the tool can be promoted to stable tooling:

- [ ] All safety checks pass
- [ ] Test suite passes consistently
- [ ] Manual testing on real codebase shows good results
- [ ] No false positives (incorrect fixes)
- [ ] Handles edge cases gracefully
- [ ] Performance is acceptable on large codebases

### 5. Integration Path

Once stable, the tool can be:
1. Moved to `tools/` directory
2. Added to `pyproject.toml` dependencies
3. Integrated into `make fix` target
4. Added to CI/CD pipeline

## Current Status

**EXPERIMENTAL** - Do not use in production workflow yet.

The tool is under development and testing. Use only for experimentation and validation.
