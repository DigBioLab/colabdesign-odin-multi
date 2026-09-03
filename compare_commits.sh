#!/bin/bash

# Usage: ./semantic-git-diff.sh <commit1> <commit2> <directory> [--vscode]
# Example (default consolidated diff): ./semantic-git-diff.sh abc1234 def5678 path/to/your/dir
# Example (open each diff in VSCode): ./semantic-git-diff.sh abc1234 def5678 path/to/your/dir --vscode

if [ $# -lt 3 ] || [ $# -gt 4 ]; then
  echo "Usage: $0 <commit1> <commit2> <directory> [--vscode]"
  exit 1
fi

COMMIT1=$1
COMMIT2=$2
DIR_PATH=$3
OPEN_VSCODE=0

if [ $# -eq 4 ]; then
  if [ "$4" == "--vscode" ]; then
    OPEN_VSCODE=1
  else
    echo "Unknown flag: $4"
    echo "Usage: $0 <commit1> <commit2> <directory> [--vscode]"
    exit 1
  fi
fi

if [ ! -d "$DIR_PATH" ]; then
  echo "Error: '$DIR_PATH' is not a directory."
  exit 1
fi

# Create a unique temporary directory
TEMP_DIR="/tmp/gitdiff_$$"  # Use PID to avoid conflicts
mkdir -p "$TEMP_DIR"

# Create the Python script that extracts the semantic structure
cat > "$TEMP_DIR/semantic_diff.py" << 'EOF'
#!/usr/bin/env python3
import ast
import sys
import difflib
import re
import astunparse
import black
from pathlib import Path

class DocstringRemover(ast.NodeTransformer):
    """AST transformer that removes all docstrings from functions, classes, and modules"""
    
    def visit_FunctionDef(self, node):
        node = self.generic_visit(node)
        if (node.body and isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Str)):
            node.body = node.body[1:]
        return node
    
    def visit_ClassDef(self, node):
        node = self.generic_visit(node)
        if (node.body and isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Str)):
            node.body = node.body[1:]
        return node
    
    def visit_Module(self, node):
        node = self.generic_visit(node)
        if (node.body and isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Str)):
            node.body = node.body[1:]
        return node
    
    def visit_AsyncFunctionDef(self, node):
        node = self.generic_visit(node)
        if (node.body and isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Str)):
            node.body = node.body[1:]
        return node

def remove_comments(code):
    code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
    return code

def normalize_code(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        code = f.read()
    
    # First remove comments
    code = remove_comments(code)
    
    try:
        tree = ast.parse(code)
        tree = DocstringRemover().visit(tree)
        normalized = astunparse.unparse(tree)
        try:
            mode = black.Mode(line_length=88, string_normalization=True)
            normalized = black.format_str(normalized, mode=mode)
        except Exception as e:
            print(f"Warning: Black formatting failed: {e}", file=sys.stderr)
        return normalized
    except SyntaxError as e:
        print(f"Warning: Could not parse {file_path} as valid Python. Falling back to basic normalization: {e}", file=sys.stderr)
        
        def preserve_newlines_in_triple_quotes(match):
            text = match.group(0)
            newline_count = text.count('\n')
            return '\n' * newline_count

        code = re.sub(r'""".*?"""', preserve_newlines_in_triple_quotes, code, flags=re.DOTALL)
        code = re.sub(r"'''.*?'''", preserve_newlines_in_triple_quotes, code, flags=re.DOTALL)
        code = re.sub(r'[ \t]+', ' ', code)
        code = re.sub(r'\n\s*\n+', '\n\n', code)
        return code

def prepare_diff_files(file1, file2, output1, output2):
    try:
        code1 = normalize_code(file1)
        code2 = normalize_code(file2)
        with open(output1, 'w', encoding='utf-8') as f:
            f.write(code1)
        with open(output2, 'w', encoding='utf-8') as f:
            f.write(code2)
        return True
    except Exception as e:
        print(f"Error processing files: {e}", file=sys.stderr)
        return False

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(f"Usage: {sys.argv[0]} <input_file1> <input_file2> <output_file1> <output_file2>")
        sys.exit(1)
    
    input_file1 = sys.argv[1]
    input_file2 = sys.argv[2]
    output_file1 = sys.argv[3]
    output_file2 = sys.argv[4]
    
    if prepare_diff_files(input_file1, input_file2, output_file1, output_file2):
        print("Files prepared successfully!")
    else:
        print("Failed to prepare files for diff.")
        sys.exit(1)
EOF

# Check and install dependencies if needed
echo "Checking for required Python packages (astunparse and black)..."
python3 -c "import astunparse, black" 2>/dev/null || {
    echo "Installing required Python packages: astunparse, black..."
    pip install astunparse black >/dev/null 2>&1 || {
        echo "Error: Failed to install required packages. Please install manually:"
        echo "  pip install astunparse black"
        exit 1
    }
    echo "Packages installed successfully!"
}

echo "Searching for .py files in $DIR_PATH..."
PY_FILES=$(find "$DIR_PATH" -type f -name "*.py")

if [ -z "$PY_FILES" ]; then
  echo "No Python files found in the directory: $DIR_PATH"
  rm -rf "$TEMP_DIR"
  exit 0
fi

echo "Processing .py files for meaningful diffs..."

if [ $OPEN_VSCODE -eq 0 ]; then
    # Initialize consolidated diff files (only used if not opening VSCode)
    CONSOLIDATED_BEFORE="/tmp/all_diffs_before.py"
    CONSOLIDATED_AFTER="/tmp/all_diffs_after.py"
    CONSOLIDATED_DIFF="./all_diffs_unified.diff"
    > "$CONSOLIDATED_BEFORE"
    > "$CONSOLIDATED_AFTER"
    > "$CONSOLIDATED_DIFF"
fi

DIFF_FOUND=0
FILES_WITH_DIFFS=()

for FILE_PATH in $PY_FILES; do
    # Ensure the file exists in both commits; if not, skip it
    if ! git cat-file -e "$COMMIT1":"$FILE_PATH" 2>/dev/null; then
        continue
    fi
    if ! git cat-file -e "$COMMIT2":"$FILE_PATH" 2>/dev/null; then
        continue
    fi

    # Create a safe filename by replacing '/' with '_'
    SAFE_FILENAME=$(echo "$FILE_PATH" | tr '/' '_')

    # Extract the two versions into temporary files
    git show "$COMMIT1":"$FILE_PATH" > "$TEMP_DIR/${SAFE_FILENAME}_orig1.py" 2>/dev/null
    git show "$COMMIT2":"$FILE_PATH" > "$TEMP_DIR/${SAFE_FILENAME}_orig2.py" 2>/dev/null

    # Normalize the files using the Python script
    python3 "$TEMP_DIR/semantic_diff.py" \
        "$TEMP_DIR/${SAFE_FILENAME}_orig1.py" \
        "$TEMP_DIR/${SAFE_FILENAME}_orig2.py" \
        "$TEMP_DIR/${SAFE_FILENAME}_normalized1.py" \
        "$TEMP_DIR/${SAFE_FILENAME}_normalized2.py"

    # Generate a unified diff for these normalized files
    diff -u \
        "$TEMP_DIR/${SAFE_FILENAME}_normalized1.py" \
        "$TEMP_DIR/${SAFE_FILENAME}_normalized2.py" \
        > "$TEMP_DIR/${SAFE_FILENAME}_unified_diff.txt" || true

    if [ -s "$TEMP_DIR/${SAFE_FILENAME}_unified_diff.txt" ]; then
        DIFF_FOUND=1
        FILES_WITH_DIFFS+=("$FILE_PATH")

        # Copy normalized versions to /tmp with consistent naming
        cp "$TEMP_DIR/${SAFE_FILENAME}_normalized1.py" "/tmp/${SAFE_FILENAME}_before.py"
        cp "$TEMP_DIR/${SAFE_FILENAME}_normalized2.py" "/tmp/${SAFE_FILENAME}_after.py"
        cp "$TEMP_DIR/${SAFE_FILENAME}_unified_diff.txt" "/tmp/${SAFE_FILENAME}_diff.txt"

        if [ $OPEN_VSCODE -eq 1 ]; then
            # Open the diff in VSCode
            code --diff "/tmp/${SAFE_FILENAME}_before.py" "/tmp/${SAFE_FILENAME}_after.py"
            echo "================================================================================"
            echo "Opened VSCode diff for file: $FILE_PATH"
            echo "  Before: /tmp/${SAFE_FILENAME}_before.py"
            echo "  After:  /tmp/${SAFE_FILENAME}_after.py"
            echo "  Diff:   /tmp/${SAFE_FILENAME}_diff.txt"
            echo "================================================================================"
            echo ""
        else
            # Append normalized versions and diff to the consolidated files
            echo -e "\n\n# ======== FILE: $FILE_PATH ========\n" >> "$CONSOLIDATED_BEFORE"
            cat "$TEMP_DIR/${SAFE_FILENAME}_normalized1.py" >> "$CONSOLIDATED_BEFORE"
            
            echo -e "\n\n# ======== FILE: $FILE_PATH ========\n" >> "$CONSOLIDATED_AFTER"
            cat "$TEMP_DIR/${SAFE_FILENAME}_normalized2.py" >> "$CONSOLIDATED_AFTER"
            
            echo -e "\n\n# ======== FILE: $FILE_PATH ========\n" >> "$CONSOLIDATED_DIFF"
            cat "$TEMP_DIR/${SAFE_FILENAME}_unified_diff.txt" >> "$CONSOLIDATED_DIFF"
            
            echo ""
            echo "================================================================================"
            echo "Semantic diff found for file: $FILE_PATH"
            echo "  Individual files saved as:"
            echo "  Before: /tmp/${SAFE_FILENAME}_before.py"
            echo "  After:  /tmp/${SAFE_FILENAME}_after.py"
            echo "  Diff:   /tmp/${SAFE_FILENAME}_diff.txt"
            echo "================================================================================"
            echo ""
        fi
    fi
done

if [ $DIFF_FOUND -eq 0 ]; then
    echo "No semantic (non-formatting) differences found in any .py files."
else
    if [ $OPEN_VSCODE -eq 0 ]; then
        echo ""
        echo "================================================================================"
        echo "Consolidated files saved to:"
        echo "  Before: $CONSOLIDATED_BEFORE"
        echo "  After:  $CONSOLIDATED_AFTER"
        echo "  Unified Diff (with - and + format): $CONSOLIDATED_DIFF (in current directory)"
        echo ""
        echo "Files with semantic differences (${#FILES_WITH_DIFFS[@]} total):"
        for file in "${FILES_WITH_DIFFS[@]}"; do
            echo "  - $file"
        done
        echo "================================================================================"
        
        echo ""
        echo "==================== DIFF CONTENT START ===================="
        cat "$CONSOLIDATED_DIFF"
        echo "==================== DIFF CONTENT END ======================"
    fi
fi

# Cleanup the temporary directory
rm -rf "$TEMP_DIR"
exit 0
