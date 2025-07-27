#!/usr/bin/env python3
"""Quick script to fix common linting issues in test files."""
import os
import re
import sys

def fix_unused_imports(file_path):
    """Remove obvious unused imports from Python test files."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Common unused imports in test files
    unused_patterns = [
        r"from unittest\.mock import.*MagicMock.*\n",
        r"from unittest\.mock import.*Mock.*\n", 
        r"from unittest\.mock import.*patch.*\n",
        r"import tempfile\n",
        r"import torch\n",
        r"import time\n",
    ]
    
    # Only remove if not used in file
    lines = content.split('\n')
    new_lines = []
    
    for line in lines:
        skip_line = False
        
        # Check for unused MagicMock import
        if "from unittest.mock import" in line and "MagicMock" in line:
            if "MagicMock" not in content.replace(line, ""):
                skip_line = True
        
        # Check for unused Mock import (but keep if MagicMock is used)
        elif "from unittest.mock import" in line and "Mock" in line and "MagicMock" not in line:
            if "Mock" not in content.replace(line, ""):
                skip_line = True
                
        # Check for unused patch import
        elif "from unittest.mock import" in line and "patch" in line:
            if "@patch" not in content and "patch(" not in content:
                skip_line = True
                
        # Check for unused tempfile
        elif "import tempfile" in line:
            if "tempfile." not in content.replace(line, ""):
                skip_line = True
                
        # Check for unused torch
        elif "import torch" in line:
            if "torch." not in content.replace(line, ""):
                skip_line = True
                
        # Check for unused time
        elif "import time" in line:
            if "time." not in content.replace(line, "") and "time.sleep" not in content.replace(line, ""):
                skip_line = True
        
        if not skip_line:
            new_lines.append(line)
    
    new_content = '\n'.join(new_lines)
    
    # Fix bare except statements
    new_content = re.sub(r'except:', 'except Exception:', new_content)
    
    # Fix unused variables by prefixing with underscore
    new_content = re.sub(r"(\s+)([a-zA-Z_][a-zA-Z0-9_]*) = ([^=\n]+) # F841", r"\1_\2 = \3", new_content)
    
    if new_content != content:
        with open(file_path, 'w') as f:
            f.write(new_content)
        print(f"Fixed imports in {file_path}")

def fix_long_lines(file_path):
    """Break long lines in Python files."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        if len(line.strip()) > 88:
            # Try to break at common patterns
            if '=' in line and len(line) > 88:
                # Break assignment lines
                if line.count('=') == 1 and line.count('==') == 0:
                    parts = line.split('=', 1)
                    if len(parts[0].strip()) < 40:
                        new_line = f"{parts[0].rstrip()} = (\n{' ' * (len(parts[0]) - len(parts[0].lstrip()))}    {parts[1].lstrip()}"
                        if not new_line.endswith(')\n'):
                            new_line = new_line.rstrip() + '\n'
                            if '(' in new_line and ')' not in new_line:
                                new_line = new_line.rstrip() + ')\n'
                        new_lines.append(new_line)
                        continue
            
            # Break long string literals
            if '"""' in line and line.count('"""') >= 2:
                new_lines.append(line)
                continue
                
            # For other long lines, just add them as-is for now
            new_lines.append(line)
        else:
            new_lines.append(line)
    
    new_content = ''.join(new_lines)
    with open(file_path, 'w') as f:
        f.write(new_content)

def main():
    """Fix linting issues in all Python test files."""
    test_dirs = ['tests/', 'examples/', 'docs/cost-optimization/', 'docs/security-configs/']
    
    for test_dir in test_dirs:
        if not os.path.exists(test_dir):
            continue
            
        for root, dirs, files in os.walk(test_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    print(f"Processing {file_path}")
                    fix_unused_imports(file_path)

if __name__ == "__main__":
    main()