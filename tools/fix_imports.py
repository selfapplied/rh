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
from typing import Dict, List, Optional, Tuple


class ImportFixer:
    """Automatically fixes broken imports by finding the correct module paths."""
    
    # Constants
    IMPORT_WORKS = "works"  # Magic string indicating import works
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.module_cache: Dict[str, Optional[str]] = {}
        self.fixes_applied = 0
        self.rename_mappings = {}  # Store symbol renames detected during analysis
        
        # Load project configuration
        self.project_config = self._load_project_config()
        
        # No whitelist - we'll test imports to see if they actually work
    
    def _load_project_config(self) -> dict:
        """Load project configuration from pyproject.toml."""
        config = {
            'first_party_modules': ['code', 'tools'],
            'project_type': 'poetry',
            'package_mode': False
        }
        
        try:
            import toml
            pyproject_path = self.project_root / "pyproject.toml"
            if pyproject_path.exists():
                with open(pyproject_path, 'r') as f:
                    data = toml.load(f)
                
                # Extract isort configuration
                if 'tool' in data and 'isort' in data['tool']:
                    isort_config = data['tool']['isort']
                    if 'known_first_party' in isort_config:
                        config['first_party_modules'] = isort_config['known_first_party']
                
                # Extract poetry configuration
                if 'tool' in data and 'poetry' in data['tool']:
                    poetry_config = data['tool']['poetry']
                    if 'package-mode' in poetry_config:
                        config['package_mode'] = poetry_config['package-mode']
                        config['project_type'] = 'poetry'
                        
        except ImportError:
            # toml not available, use defaults
            pass
        except Exception:
            # Any other error, use defaults
            pass
            
        return config
    
    def _ensure_code_path_in_sys_path(self):
        """Ensure the first-party directories are in sys.path for import testing."""
        first_party = self.project_config.get('first_party_modules', ['code', 'tools'])
        for module in first_party:
            module_path = self.project_root / module
            if module_path.exists() and str(module_path) not in sys.path:
                sys.path.insert(0, str(module_path))
    
    def _build_grep_command(self, pattern: str, search_dir: Path) -> List[str]:
        """Build a grep command for finding function/class definitions."""
        return [
            "grep", "-rn", "-A", "2",
            pattern,
            str(search_dir)
        ]
        
    def can_import(self, module_name: str, file_path: Optional[Path] = None) -> bool:
        """Test if an import actually works."""
        try:
            self._ensure_code_path_in_sys_path()
            importlib.import_module(module_name)
            return True
        except (ImportError, ModuleNotFoundError, ValueError):
            # For relative imports, also check if the file exists
            if module_name.startswith('.'):
                if file_path:
                    # Convert relative import to file path
                    relative_path = module_name.lstrip('.')
                    if relative_path:
                        # Handle relative import like '.pascal' -> 'pascal.py'
                        target_file = file_path.parent / f"{relative_path}.py"
                    else:
                        # Handle relative import like '.' -> '__init__.py'
                        target_file = file_path.parent / "__init__.py"
                    
                    if target_file.exists():
                        return True
            return False
    
    def can_import_symbol(self, module_name: str, symbol_name: str) -> bool:
        """Test if a specific symbol can be imported from a module."""
        try:
            self._ensure_code_path_in_sys_path()
            module = importlib.import_module(module_name)
            return hasattr(module, symbol_name)
        except (ImportError, ModuleNotFoundError, ValueError):
            return False
        
    def is_safe_to_modify(self, file_path: Path) -> bool:
        """Check if a file is safe to modify (clean or untracked only)."""
        try:
            # Get relative path from project root
            rel_path = file_path.relative_to(self.project_root)
            
            # Check git status for this specific file
            result = subprocess.run([
                "git", "status", "--porcelain", str(rel_path)
            ], capture_output=True, text=True, cwd=self.project_root, timeout=10)
            
            if result.returncode != 0:
                # If git command fails, assume file is safe (not in git)
                return True
            
            # Check the status
            status_lines = result.stdout.strip().split('\n')
            for line in status_lines:
                if line.strip():
                    status = line[0:2].strip()
                    if status in ['M', 'D', 'R', 'C']:  # Modified, Deleted, Renamed, or Copied
                        return False  # File has uncommitted changes - NOT safe
                    elif status == '??':
                        return True   # Untracked file - safe to modify
                    elif status == ' ':
                        return True   # Clean file - safe to modify
            
            # If no status found, assume safe
            return True
        except Exception:
            # If we can't check, assume file is safe
            return True
    
    
    
    
    def has_broken_imports(self, file_path: Path) -> bool:
        """Check if a file has broken imports."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception:
            return False
        
        return self.has_broken_imports_from_content(content, file_path)
    
    def check_safety(self) -> bool:
        """Check if repository is safe to modify (allows untracked files)."""
        try:
            # Check if we're in a git repository
            git_dir = self.project_root / ".git"
            if not git_dir.exists():
                print("⚠️  Warning: Not in a git repository - proceeding with caution")
                return True
            
            # Check git status to see what changes exist
            result = subprocess.run([
                "git", "status", "--porcelain"
            ], capture_output=True, text=True, cwd=self.project_root, timeout=10)
            
            if result.returncode != 0:
                print("⚠️  Warning: Could not check git status - proceeding anyway")
                return True
            
            if result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                tracked_changes = [line for line in lines if not line.startswith('??')]
                untracked_files = [line for line in lines if line.startswith('??')]
                
                if tracked_changes:
                    print("⚠️  Warning: You have uncommitted changes to tracked files:")
                    for line in tracked_changes[:5]:  # Show first 5
                        print(f"     {line}")
                    if len(tracked_changes) > 5:
                        print(f"     ... and {len(tracked_changes) - 5} more")
                    print("   The import fixer may modify these files. Consider committing first.")
                    
                    # Ask for confirmation (non-interactive mode just warns)
                    print("   Proceeding anyway...")
                
                if untracked_files:
                    print(f"ℹ️  Found {len(untracked_files)} untracked files - these are safe to modify")
                
                return True
            else:
                print("✅ Repository is clean and safe to modify")
                return True
            
        except subprocess.TimeoutExpired:
            print("⚠️  Warning: Git command timed out - proceeding anyway")
            return True
        except Exception as e:
            print(f"⚠️  Warning: Error checking repository ({e}) - proceeding anyway")
            return True
        
    def find_module(self, module_name: str, file_path: Optional[Path] = None) -> Optional[str]:
        """Find the actual path of a module."""
        if module_name in self.module_cache:
            return self.module_cache[module_name]
        
        # First, test if the import actually works as-is
        if self.can_import(module_name, file_path):
            self.module_cache[module_name] = self.IMPORT_WORKS
            return self.IMPORT_WORKS
        
        # Only try to fix imports that actually fail
        print(f"  Import '{module_name}' failed - attempting to find alternative...")
        
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
                root_dir / f"{module_name.replace('.', '/')}.py",
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
                cmd = self._build_grep_command(
                    f"\\(def\\|class\\) {re.escape(function_name)}",
                    search_dir
                )
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
        """Test if an import fix would work by checking if the module path exists."""
        try:
            # For relative imports, check if the target file exists
            if new_import.startswith('.'):
                # Convert relative import to file path
                relative_path = new_import.lstrip('.')
                if relative_path:
                    target_file = file_path.parent / f"{relative_path}.py"
                else:
                    target_file = file_path.parent / "__init__.py"
                return target_file.exists()
            else:
                # For absolute imports, ensure code directory is in path and try to import
                self._ensure_code_path_in_sys_path()
                importlib.import_module(new_import)
                return True
        except (ImportError, ModuleNotFoundError, ValueError):
            return False
    
    def find_correct_import_path(self, broken_import: str, imported_items: List[str], file_path: Path) -> Optional[str]:
        """Find the correct import path by grepping for the imported items."""
        print(f"  Analyzing broken import: {broken_import}")
        print(f"  Looking for items: {imported_items}")
        
        # Use first-party modules from project configuration
        first_party = self.project_config.get('first_party_modules', ['code', 'tools'])
        search_dirs = [
            self.project_root / module for module in first_party 
            if (self.project_root / module).exists()
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
            print(f"    🔍 Attempting to find renamed symbols...")
            return self.find_renamed_symbols(broken_import, imported_items, file_path, search_dirs)
        
        # Find the most common file for all items
        file_counts = {}
        for item, locations in item_locations.items():
            for file_path, _, _, _ in locations:
                file_counts[file_path] = file_counts.get(file_path, 0) + 1
        
        if not file_counts:
            return None
        
        # Get candidates sorted by frequency
        candidates = sorted(file_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Test each candidate to see if it resolves the import
        for candidate_file, count in candidates:
            suggested_module = self._build_module_name(candidate_file)
            if not suggested_module:
                continue
                
            print(f"    Testing candidate: {suggested_module} (found {count} matches)")
            
            # Test if this import fix works
            if self.test_import_fix(file_path, broken_import, suggested_module):
                print(f"    ✅ Import test passed: {suggested_module}")
                return suggested_module
            else:
                print(f"    ❌ Import test failed: {suggested_module}")
        
        # If no candidate passed the test, return the most frequent one anyway
        best_file = candidates[0][0]
        suggested_module = self._build_module_name(best_file)
        if suggested_module:
            print(f"    ⚠️  Using best guess: {suggested_module} (no test validation)")
            return suggested_module
        
        return None
    
    def _build_module_name(self, file_path: Path) -> Optional[str]:
        """Build a module name from a file path based on project configuration."""
        rel_path = file_path.relative_to(self.project_root)
        module_parts = list(rel_path.parts)
        
        # Remove .py extension if present
        if module_parts[-1].endswith('.py'):
            module_parts[-1] = module_parts[-1][:-3]
        
        # Handle __init__.py case
        if module_parts[-1] == '__init__':
            module_parts = module_parts[:-1]
        
        # Handle first-party modules based on project configuration
        first_party = self.project_config.get('first_party_modules', ['code', 'tools'])
        if module_parts and module_parts[0] in first_party:
            # For first-party modules, remove the prefix since we add the directory to sys.path
            # This makes 'code/riemann/ce1.py' become 'riemann.ce1'
            module_parts = module_parts[1:]
        
        return ".".join(module_parts) if module_parts else None
    
    def break_symbol_into_atoms(self, symbol: str) -> List[str]:
        """Break a symbol name into meaningful atoms for fuzzy matching."""
        import re

        # Split on common separators and camelCase
        atoms = []
        
        # Handle snake_case first: compute_zeta_values -> [compute, zeta, values]
        snake_split = symbol.split('_')
        for part in snake_split:
            if part and len(part) > 1:
                # Further split camelCase within snake_case parts
                camel_parts = re.split(r'(?=[A-Z])', part)
                for camel_part in camel_parts:
                    if camel_part and len(camel_part) > 1:
                        atoms.append(camel_part.lower())
        
        # Handle pure camelCase: PascalExplicitFormula -> [Pascal, Explicit, Formula]
        if not atoms:  # Only if snake_case didn't produce anything
            camel_split = re.split(r'(?=[A-Z])', symbol)
            for part in camel_split:
                if part and len(part) > 1:  # Avoid single letters
                    atoms.append(part.lower())
        
        # Handle common prefixes/suffixes - be more conservative
        common_suffixes = ['formula', 'function', 'class', 'method', 'algorithm', 'analyzer', 'processor', 'main']
        for suffix in common_suffixes:
            if suffix in atoms and len(atoms) > 1:  # Only remove if there are other atoms
                atoms.remove(suffix)
        
        # Filter out very short atoms and common words
        filtered_atoms = []
        for atom in atoms:
            if len(atom) > 2 and atom not in ['def', 'class', 'main', 'test', 'run', 'get', 'set']:
                filtered_atoms.append(atom)
        
        return filtered_atoms
    
    def extract_signature_info(self, signature: str) -> dict:
        """Extract type signature information for compatibility matching."""
        import re
        
        info = {
            'is_class': 'class ' in signature,
            'is_function': 'def ' in signature,
            'has_params': '(' in signature and ')' in signature,
            'param_count': 0,
            'has_return_type': '->' in signature,
            'has_complex_types': False
        }
        
        if info['has_params']:
            # Count parameters (rough estimate)
            param_part = signature.split('(')[1].split(')')[0]
            if param_part.strip():
                # Count commas + 1, but be careful about nested structures
                info['param_count'] = param_part.count(',') + 1
        
        # Check for complex types (List, Dict, Optional, etc.)
        complex_type_patterns = [
            r'List\[', r'Dict\[', r'Optional\[', r'Tuple\[', 
            r'Union\[', r'Set\[', r'Callable\['
        ]
        for pattern in complex_type_patterns:
            if re.search(pattern, signature):
                info['has_complex_types'] = True
                break
        
        return info
    
    def calculate_symbol_similarity(self, original: str, candidate: str, original_sig: str, candidate_sig: str) -> float:
        """Calculate similarity score between original and candidate symbols."""
        original_atoms = set(self.break_symbol_into_atoms(original))
        candidate_atoms = set(self.break_symbol_into_atoms(candidate))
        
        if not original_atoms or not candidate_atoms:
            return 0.0
        
        # Jaccard similarity of atoms
        intersection = len(original_atoms & candidate_atoms)
        union = len(original_atoms | candidate_atoms)
        atom_similarity = intersection / union if union > 0 else 0.0
        
        # Signature compatibility bonus
        sig_bonus = 0.0
        original_info = self.extract_signature_info(original_sig)
        candidate_info = self.extract_signature_info(candidate_sig)
        
        # Type compatibility (class vs function)
        if original_info['is_class'] == candidate_info['is_class']:
            sig_bonus += 0.2
        
        # Parameter count similarity
        if original_info['param_count'] == candidate_info['param_count']:
            sig_bonus += 0.1
        elif abs(original_info['param_count'] - candidate_info['param_count']) <= 1:
            sig_bonus += 0.05
        
        # Complex types similarity
        if original_info['has_complex_types'] == candidate_info['has_complex_types']:
            sig_bonus += 0.1
        
        return atom_similarity + sig_bonus
    
    def find_renamed_symbols(self, broken_import: str, imported_items: List[str], file_path: Path, search_dirs: List[Path] = None) -> Optional[str]:
        """Find symbols that might have been renamed by analyzing atoms and signatures."""
        print(f"    🔍 Searching for renamed symbols...")
        
        # Get all function/class definitions in the codebase
        all_definitions = []
        if search_dirs is None:
            # Use first-party modules from project configuration
            first_party = self.project_config.get('first_party_modules', ['code', 'tools'])
            search_dirs = [
                self.project_root / module for module in first_party 
                if (self.project_root / module).exists()
            ]
        
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            
            try:
                # Find all class and function definitions
                cmd = self._build_grep_command(
                    "\\(def\\|class\\) [a-zA-Z_][a-zA-Z0-9_]*",
                    search_dir
                )
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    i = 0
                    while i < len(lines):
                        line = lines[i]
                        if line and ':' in line:
                            # Debug: print lines being processed
                            if 'pascal' in line.lower():
                                print(f"      🔍 Processing Pascal line: {line}")
                            parts = line.split(':', 2)
                            if len(parts) >= 3:
                                file_path_obj = Path(parts[0])
                                line_num = int(parts[1])
                                content = parts[2]
                                
                                # Extract symbol name and signature
                                if 'def ' in content:
                                    symbol_match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)', content)
                                elif 'class ' in content:
                                    symbol_match = re.search(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)', content)
                                else:
                                    symbol_match = None
                                
                                if symbol_match:
                                    symbol_name = symbol_match.group(1)
                                    # Get full signature
                                    signature_lines = [content]
                                    for j in range(1, min(3, len(lines) - i)):
                                        if i + j < len(lines) and lines[i + j].strip():
                                            signature_lines.append(lines[i + j].strip())
                                    
                                    full_signature = ' '.join(signature_lines)
                                    all_definitions.append((file_path_obj, line_num, symbol_name, full_signature))
                                    
                                    # Debug: print when we find pascal-related functions
                                    if 'pascal' in symbol_name.lower():
                                        rel_path = file_path_obj.relative_to(self.project_root)
                                        print(f"      🔍 Found Pascal function: {rel_path}:{line_num} - {symbol_name}")
                        i += 1
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError):
                continue
        
        print(f"    Found {len(all_definitions)} total definitions to analyze")
        
        # Debug: show some of the definitions found
        for i, (file_path_obj, line_num, symbol_name, signature) in enumerate(all_definitions[:5]):
            rel_path = file_path_obj.relative_to(self.project_root)
            print(f"      {i+1}. {rel_path}:{line_num} - {symbol_name}")
        if len(all_definitions) > 5:
            print(f"      ... and {len(all_definitions) - 5} more")
        
        # Debug: check if calculate_pascal_brackets is in the list
        pascal_functions = [(f, l, s, sig) for f, l, s, sig in all_definitions if 'pascal' in s.lower()]
        print(f"    Pascal-related functions found: {len(pascal_functions)}")
        for file_path_obj, line_num, symbol_name, signature in pascal_functions:
            rel_path = file_path_obj.relative_to(self.project_root)
            print(f"      {rel_path}:{line_num} - {symbol_name}")
        
        # Find best matches for each imported item
        best_matches = {}
        for item in imported_items:
            best_match = None
            best_score = 0.0
            
            for file_path_obj, line_num, symbol_name, signature in all_definitions:
                score = self.calculate_symbol_similarity(item, symbol_name, "", signature)
                if score > best_score and score > 0.6:  # Higher threshold for confidence
                    best_score = score
                    best_match = (file_path_obj, line_num, symbol_name, signature, score)
            
            if best_match:
                best_matches[item] = best_match
                print(f"    🎯 Best match for '{item}': '{best_match[2]}' (score: {best_match[4]:.2f})")
                rel_path = best_match[0].relative_to(self.project_root)
                print(f"      {rel_path}:{best_match[1]} - {best_match[3]}")
        
        if not best_matches:
            print(f"    ❌ No confident matches found for renamed symbols")
            return None
        
        # Find the most common directory for the best matches
        dir_counts = {}
        for item, (file_path_obj, _, _, _, _) in best_matches.items():
            dir_path = file_path_obj.parent
            dir_counts[dir_path] = dir_counts.get(dir_path, 0) + 1
        
        if not dir_counts:
            return None
        
        # Get the best directory
        best_dir = max(dir_counts.items(), key=lambda x: x[1])[0]
        suggested_module = self._build_module_name(best_dir)
        
        if not suggested_module:
            return None
            
        print(f"    ✅ Confident rename detected! Module: {suggested_module}")
        print(f"    📝 Consider updating import to use the new symbol names")
        
        # Store the rename mappings for later use
        self.rename_mappings = {}
        for item, (_, _, new_name, _, _) in best_matches.items():
            if item != new_name:
                self.rename_mappings[item] = new_name
                print(f"    🔄 '{item}' → '{new_name}'")
        
        return suggested_module
    
    def get_relative_import_path(self, from_file: Path, to_module: str) -> Optional[str]:
        """Convert absolute import to relative import path."""
        module_path = self.find_module(to_module)
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
    
    def fix_file(self, file_path: Path) -> bool:
        """Fix imports in a single file."""
        print(f"    Processing {file_path}")
        # Check if file is safe to modify (clean or untracked only)
        safe_to_modify = self.is_safe_to_modify(file_path)
        print(f"    File is safe to modify: {safe_to_modify}")
        if not safe_to_modify:
            print(f"⚠️  Skipping {file_path} - has uncommitted changes")
            return False
        
        # Clear rename mappings for each new file
        self.rename_mappings = {}
        
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
            if isinstance(node, ast.ImportFrom) and node.module:
                if self._process_import_from(node, file_path, new_content):
                    modified = True
            elif isinstance(node, ast.Import):
                # Handle regular imports - for now, just leave working imports alone
                pass
        
        if modified:
            # Validate that changes are only import-related
            if not self.only_imports_changed(content, '\n'.join(new_content), file_path):
                print(f"⚠️  Skipping {file_path} - changes would affect non-import code")
                return False
            
            # Test if the new imports actually work
            if not self.imports_work('\n'.join(new_content), file_path):
                print(f"⚠️  Skipping {file_path} - proposed fixes would still be broken")
                return False
            
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(new_content))
                print(f"✅ Fixed imports in {file_path}")
                return True
            except Exception as e:
                print(f"Error writing {file_path}: {e}")
                return False
                
        return False
    
    def _process_import_from(self, node: ast.ImportFrom, file_path: Path, new_content: List[str]) -> bool:
        """Process a single ImportFrom node and return True if modified."""
        if not node.module:
            return False
            
        module_path = self.find_module(node.module, file_path)
        imported_items = [alias.name for alias in node.names if alias.name]
        
        # Handle working imports
        if module_path == self.IMPORT_WORKS:
            return self._handle_working_import(node, imported_items, file_path, new_content)
        
        # Handle broken imports
        if not module_path and imported_items:
            return self._handle_broken_import(node, imported_items, file_path, new_content)
        
        return False
    
    def _handle_working_import(self, node: ast.ImportFrom, imported_items: List[str], file_path: Path, new_content: List[str]) -> bool:
        """Handle an import where the module works but symbols might not."""
        if not node.module or not imported_items:
            return False  # No symbols to check, import works fine
        
        # Test if all symbols can be imported
        all_symbols_work = all(
            self.can_import_symbol(node.module, item) 
            for item in imported_items
        )
        
        if all_symbols_work:
            return False  # Import works fine, no changes needed
        
        # Module exists but symbols don't - trigger rename detection
        print(f"\nSymbol import failure detected in {file_path}:")
        print(f"  Module '{node.module}' exists but symbols {imported_items} not found")
        
        suggested_module = self.find_correct_import_path(node.module, imported_items, file_path)
        if not suggested_module:
            return False
        
        return self._apply_import_fix(node.module, suggested_module, new_content)
    
    def _handle_broken_import(self, node: ast.ImportFrom, imported_items: List[str], file_path: Path, new_content: List[str]) -> bool:
        """Handle a completely broken import."""
        if not node.module:
            return False
            
        print(f"\nBroken import detected in {file_path}:")
        
        suggested_module = self.find_correct_import_path(node.module, imported_items, file_path)
        if not suggested_module:
            print(f"    ❌ Could not find suitable replacement for {node.module}")
            return False
        
        return self._apply_import_fix(node.module, suggested_module, new_content)
    
    def _apply_import_fix(self, old_module: str, new_module: str, new_content: List[str]) -> bool:
        """Apply an import fix by replacing the module name and symbols."""
        for i, line in enumerate(new_content):
            if f"from {old_module}" in line:
                new_line = line.replace(f"from {old_module}", f"from {new_module}")
                
                # Apply symbol renames if we detected any
                if self.rename_mappings:
                    for old_symbol, new_symbol in self.rename_mappings.items():
                        if old_symbol in new_line:
                            new_line = new_line.replace(old_symbol, new_symbol)
                            print(f"    🔄 Applied symbol rename: '{old_symbol}' → '{new_symbol}'")
                
                new_content[i] = new_line
                self.fixes_applied += 1
                print(f"    ✅ Fixed: {old_module} -> {new_module}")
                return True
        
        return False
    
    def imports_work(self, content: str, file_path: Path) -> bool:
        """Test if all imports in the content actually work."""
        return not self.has_broken_imports_from_content(content, file_path)
    
    def has_broken_imports_from_content(self, content: str, file_path: Path) -> bool:
        """Check if content has broken imports (reusable version of has_broken_imports)."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return True  # Syntax error means broken imports
            
        # Check each import statement
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module:  # Any import (absolute or relative)
                    # Test if the module can be imported
                    if not self.can_import(node.module, file_path):
                        return True
                    # Test specific symbols if any
                    if node.names:
                        for alias in node.names:
                            if alias.name and not self.can_import_symbol(node.module, alias.name):
                                return True
            elif isinstance(node, ast.Import):
                # Test regular imports
                for alias in node.names:
                    if alias.name and not self.can_import(alias.name, file_path):
                        return True
        
        return False
    
    def only_imports_changed(self, original_content: str, new_content: str, file_path: Path) -> bool:
        """Validate that changes are only import-related."""
        import difflib

        # Get the diff
        original_lines = original_content.splitlines()
        new_lines = new_content.splitlines()
        
        # Use unified_diff to get line-by-line changes
        diff = list(difflib.unified_diff(original_lines, new_lines, 
                                       fromfile='original', tofile='new', lineterm=''))
        
        # Check each change
        for line in diff:
            if line.startswith('@@'):  # Skip hunk headers
                continue
            elif line.startswith('---') or line.startswith('+++'):  # Skip file headers
                continue
            elif line.startswith('-'):  # Removed line
                removed_line = line[1:].strip()
                if not self.is_import(removed_line):
                    print(f"    ❌ Non-import change detected: removed '{removed_line}'")
                    return False
            elif line.startswith('+'):  # Added line
                added_line = line[1:].strip()
                if not self.is_import(added_line):
                    print(f"    ❌ Non-import change detected: added '{added_line}'")
                    return False
        
        return True
    
    def is_import(self, line: str) -> bool:
        """Check if a line is an import statement."""
        stripped = line.strip()
        return (stripped.startswith('import ') or 
                stripped.startswith('from ') and ' import ' in stripped)
    
    def fix_imports(self, target_path: str) -> int:
        """Fix imports in a file or directory."""
        # Safety check first (now just warns, doesn't block)
        self.check_safety()
        
        path = Path(target_path).resolve()
        
        if path.is_file() and path.suffix == '.py':
            files = [path]
        elif path.is_dir():
            # Exclude virtual environments and other unnecessary directories
            exclude_dirs = {'.venv', 'venv', '.env', 'env', '__pycache__', '.git', 'node_modules', '.pytest_cache'}
            files = []
            for py_file in path.rglob("*.py"):
                # Skip files in excluded directories
                if any(part in exclude_dirs for part in py_file.parts):
                    continue
                # Skip files in hidden directories (starting with .)
                if any(part.startswith('.') and part not in {'.', '..'} for part in py_file.parts):
                    continue
                files.append(py_file)
        else:
            print(f"Invalid path: {path}")
            return 0
            
        fixed_files = 0
        for file_path in files:
            if self.fix_file(file_path):
                fixed_files += 1
                
        return fixed_files


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix broken imports in Python files")
    parser.add_argument("path", nargs="?", default="code", help="File or directory to process")
    
    args = parser.parse_args()
    
    fixer = ImportFixer()
    print(f"Fixing imports in: {args.path}")
    
    fixed_files = fixer.fix_imports(args.path)
    print(f"\nFixed imports in {fixed_files} files")
    print(f"Total fixes applied: {fixer.fixes_applied}")


if __name__ == "__main__":
    main()
