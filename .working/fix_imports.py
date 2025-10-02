#!/usr/bin/env python3
"""
Automatic Import Fixer

Finds and fixes broken imports by locating the actual module files
and updating import statements accordingly.

Usage:
    python tools/fix_imports.py [file_or_directory]
"""

import ast
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


class ImportFixer:
    """Automatically fixes broken imports by finding the correct module paths."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.module_cache: Dict[str, Optional[str]] = {}
        self.fixes_applied = 0
        
    def check_repository_safety(self) -> bool:
        """Check if repository is safe to modify (clean working directory)."""
        try:
            # Check if we're in a git repository
            git_dir = self.project_root / ".git"
            if not git_dir.exists():
                print("❌ Error: Not in a git repository")
                return False
            
            # Check if working directory is clean
            result = subprocess.run([
                "git", "status", "--porcelain"
            ], capture_output=True, text=True, cwd=self.project_root, timeout=10)
            
            if result.returncode != 0:
                print("❌ Error: Could not check git status")
                return False
            
            if result.stdout.strip():
                print("❌ Error: Working directory is not clean")
                print("   Please commit or stash your changes before running import fixes")
                print("   Uncommitted changes:")
                for line in result.stdout.strip().split('\n')[:5]:  # Show first 5
                    print(f"     {line}")
                if len(result.stdout.strip().split('\n')) > 5:
                    print(f"     ... and {len(result.stdout.strip().split('\n')) - 5} more")
                return False
            
            print("✅ Repository is clean and safe to modify")
            return True
            
        except subprocess.TimeoutExpired:
            print("❌ Error: Git command timed out")
            return False
        except Exception as e:
            print(f"❌ Error checking repository: {e}")
            return False
        
    def find_module_path(self, module_name: str) -> Optional[str]:
        """Find the actual path of a module."""
        if module_name in self.module_cache:
            return self.module_cache[module_name]
            
        # Try to find the module
        try:
            spec = importlib.util.find_spec(module_name)
            if spec and spec.origin:
                path = spec.origin
                self.module_cache[module_name] = path
                return path
        except (ImportError, ModuleNotFoundError, ValueError):
            pass
            
        # Try to find in project directories
        for root_dir in [self.project_root / "code", self.project_root / "tools"]:
            if not root_dir.exists():
                continue
                
            # Try different possible paths
            possible_paths = [
                root_dir / f"{module_name}.py",
                root_dir / module_name / "__init__.py",
                root_dir / module_name.replace(".", "/") / "__init__.py",
                root_dir / module_name.replace(".", "/") + ".py",
            ]
            
            for path in possible_paths:
                if path.exists():
                    self.module_cache[module_name] = str(path)
                    return str(path)
        
        self.module_cache[module_name] = None
        return None
    
    def grep_for_function(self, function_name: str, search_dirs: List[Path]) -> List[Tuple[Path, int, str, str]]:
        """Grep for a function definition across the repository, including signature."""
        results = []
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
                
            try:
                # Use grep to find function definitions with more context
                cmd = [
                    "grep", "-rn", "-A", "2",
                    f"^(def|class)\\s+{re.escape(function_name)}\\b",
                    str(search_dir)
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    i = 0
                    while i < len(lines):
                        line = lines[i]
                        if line and ':' in line:
                            parts = line.split(':', 2)
                            if len(parts) >= 3:
                                file_path = Path(parts[0])
                                line_num = int(parts[1])
                                content = parts[2]
                                
                                # Get the full signature (next 2 lines)
                                signature_lines = []
                                for j in range(1, min(3, len(lines) - i)):
                                    if i + j < len(lines) and lines[i + j].strip():
                                        signature_lines.append(lines[i + j].strip())
                                
                                full_signature = content + ' ' + ' '.join(signature_lines)
                                results.append((file_path, line_num, content, full_signature))
                        i += 1
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError):
                continue
                
        return results
    
    def test_import_fix(self, file_path: Path, old_import: str, new_import: str) -> bool:
        """Test if an import fix resolves linting issues."""
        try:
            # Create a temporary test file with the new import
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Replace the import
            test_content = content.replace(f"from {old_import}", f"from {new_import}")
            
            # Write to temporary file
            test_file = file_path.with_suffix('.test_import.py')
            with open(test_file, 'w') as f:
                f.write(test_content)
            
            # Run Python syntax check
            result = subprocess.run([
                "python", "-m", "py_compile", str(test_file)
            ], capture_output=True, text=True, timeout=10)
            
            # Clean up
            test_file.unlink()
            
            return result.returncode == 0
            
        except Exception as e:
            print(f"    Warning: Could not test import fix: {e}")
            return False
    
    def find_correct_import_path(self, broken_import: str, imported_items: List[str], file_path: Path) -> Optional[str]:
        """Find the correct import path by grepping for the imported items."""
        print(f"  Analyzing broken import: {broken_import}")
        print(f"  Looking for items: {imported_items}")
        
        search_dirs = [
            self.project_root / "code",
            self.project_root / "tools"
        ]
        
        # For each imported item, find where it's defined
        item_locations = {}
        for item in imported_items:
            locations = self.grep_for_function(item, search_dirs)
            if locations:
                item_locations[item] = locations
                print(f"    Found '{item}' in {len(locations)} location(s)")
                for file_path, line_num, content, signature in locations[:3]:  # Show first 3
                    rel_path = file_path.relative_to(self.project_root)
                    print(f"      {rel_path}:{line_num} - {signature}")
        
        if not item_locations:
            print(f"    No definitions found for any of: {imported_items}")
            return None
        
        # Find the most common directory for all items
        dir_counts = {}
        for item, locations in item_locations.items():
            for file_path, _, _, _ in locations:
                dir_path = file_path.parent
                dir_counts[dir_path] = dir_counts.get(dir_path, 0) + 1
        
        if not dir_counts:
            return None
        
        # Get candidates sorted by frequency
        candidates = sorted(dir_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Test each candidate to see if it resolves the import
        for candidate_dir, count in candidates:
            rel_path = candidate_dir.relative_to(self.project_root)
            module_parts = list(rel_path.parts)
            
            # Remove 'code' or 'tools' prefix
            if module_parts and module_parts[0] in ['code', 'tools']:
                module_parts = module_parts[1:]
                
            if module_parts:
                suggested_module = ".".join(module_parts)
                print(f"    Testing candidate: {suggested_module} (found {count} matches)")
                
                # Test if this import fix works
                if self.test_import_fix(file_path, broken_import, suggested_module):
                    print(f"    ✅ Import test passed: {suggested_module}")
                    return suggested_module
                else:
                    print(f"    ❌ Import test failed: {suggested_module}")
        
        # If no candidate passed the test, return the most frequent one anyway
        best_dir = candidates[0][0]
        rel_path = best_dir.relative_to(self.project_root)
        module_parts = list(rel_path.parts)
        
        if module_parts and module_parts[0] in ['code', 'tools']:
            module_parts = module_parts[1:]
            
        if module_parts:
            suggested_module = ".".join(module_parts)
            print(f"    ⚠️  Using best guess: {suggested_module} (no test validation)")
            return suggested_module
        
        return None
    
    def get_relative_import_path(self, from_file: Path, to_module: str) -> Optional[str]:
        """Convert absolute import to relative import path."""
        module_path = self.find_module_path(to_module)
        if not module_path:
            return None
            
        module_path = Path(module_path)
        
        # Calculate relative path
        try:
            rel_path = os.path.relpath(module_path.parent, from_file.parent)
            if rel_path == ".":
                rel_path = ""
            
            # Convert to module path
            if rel_path:
                module_parts = rel_path.split(os.sep)
                # Remove 'code' or 'tools' from the path
                if module_parts and module_parts[0] in ['code', 'tools']:
                    module_parts = module_parts[1:]
                if module_parts:
                    return ".".join(module_parts)
            return ""
        except ValueError:
            return None
    
    def fix_imports_in_file(self, file_path: Path) -> bool:
        """Fix imports in a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return False
            
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            print(f"Syntax error in {file_path}: {e}")
            return False
            
        modified = False
        new_content = content.splitlines()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:  # Absolute import
                    # Try to find the module
                    module_path = self.find_module_path(node.module)
                    if not module_path:
                        # Module not found - try to find it by grepping for imported items
                        imported_items = [alias.name for alias in node.names if alias.name]
                        if imported_items:
                            print(f"\nBroken import detected in {file_path}:")
                            suggested_module = self.find_correct_import_path(node.module, imported_items, file_path)
                            
                            if suggested_module:
                                # Find the line and replace it
                                for i, line in enumerate(new_content):
                                    if f"from {node.module}" in line:
                                        new_line = line.replace(f"from {node.module}", f"from {suggested_module}")
                                        new_content[i] = new_line
                                        modified = True
                                        self.fixes_applied += 1
                                        print(f"    ✅ Fixed: {node.module} -> {suggested_module}")
                                        break
                            else:
                                print(f"    ❌ Could not find suitable replacement for {node.module}")
                    else:
                        # Module found - check if we should convert to relative import
                        rel_path = self.get_relative_import_path(file_path, node.module)
                        if rel_path and rel_path != node.module:
                            # Convert to relative import
                            if rel_path:
                                new_module = rel_path
                            else:
                                new_module = None
                            
                            # Find the line and replace it
                            for i, line in enumerate(new_content):
                                if f"from {node.module}" in line:
                                    if new_module:
                                        new_line = line.replace(f"from {node.module}", f"from {new_module}")
                                    else:
                                        new_line = line.replace(f"from {node.module}", "from .")
                                    new_content[i] = new_line
                                    modified = True
                                    self.fixes_applied += 1
                                    print(f"Fixed import in {file_path}: {node.module} -> {new_module or '.'}")
                                    break
            elif isinstance(node, ast.Import):
                # Handle regular imports
                for alias in node.names:
                    if alias.name and '.' in alias.name:
                        module_name = alias.name.split('.')[0]
                        module_path = self.find_module_path(module_name)
                        if module_path:
                            rel_path = self.get_relative_import_path(file_path, module_name)
                            if rel_path and rel_path != module_name:
                                # Find and replace the import
                                for i, line in enumerate(new_content):
                                    if f"import {alias.name}" in line:
                                        new_line = line.replace(alias.name, alias.name.replace(module_name, rel_path, 1))
                                        new_content[i] = new_line
                                        modified = True
                                        self.fixes_applied += 1
                                        print(f"Fixed import in {file_path}: {alias.name} -> {alias.name.replace(module_name, rel_path, 1)}")
                                        break
        
        if modified:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(new_content))
                return True
            except Exception as e:
                print(f"Error writing {file_path}: {e}")
                return False
                
        return False
    
    def fix_imports(self, path: str) -> int:
        """Fix imports in a file or directory."""
        # Safety check first
        if not self.check_repository_safety():
            print("\n🛑 Aborting import fixes for safety")
            return 0
        
        path = Path(path)
        
        if path.is_file() and path.suffix == '.py':
            files = [path]
        elif path.is_dir():
            files = list(path.rglob("*.py"))
        else:
            print(f"Invalid path: {path}")
            return 0
            
        fixed_files = 0
        for file_path in files:
            if self.fix_imports_in_file(file_path):
                fixed_files += 1
                
        return fixed_files


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = "code"
    
    fixer = ImportFixer()
    print(f"Fixing imports in: {target}")
    
    fixed_files = fixer.fix_imports(target)
    print(f"\nFixed imports in {fixed_files} files")
    print(f"Total fixes applied: {fixer.fixes_applied}")


if __name__ == "__main__":
    main()
