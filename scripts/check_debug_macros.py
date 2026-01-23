#!/usr/bin/env python3
"""
Check for debug macros (todo!(), unimplemented!(), dbg!()) in production Rust code.

These macros are fine in tests but should not be in production code as they
either panic (todo!, unimplemented!) or produce debug output (dbg!).

Exit codes:
  0 - No issues found
  1 - Debug macros found in production code
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple

# Patterns to detect debug macros
DEBUG_MACRO_PATTERNS = [
    (r'\btodo!\s*\(', 'todo!()'),
    (r'\bunimplemented!\s*\(', 'unimplemented!()'),
    (r'\bdbg!\s*\(', 'dbg!()'),
]

# Compiled patterns
COMPILED_PATTERNS = [(re.compile(p), name) for p, name in DEBUG_MACRO_PATTERNS]


def is_test_context(line: str, lines: List[str], line_num: int) -> bool:
    """Check if the line is within a test context."""
    # Check if line itself has #[test] or #[cfg(test)]
    if '#[test]' in line or '#[cfg(test)]' in line:
        return True

    # Look backwards for test attributes or test module
    for i in range(max(0, line_num - 20), line_num):
        prev_line = lines[i]
        if '#[cfg(test)]' in prev_line:
            return True
        if 'mod tests' in prev_line or 'mod test' in prev_line:
            return True
        if '#[test]' in prev_line:
            return True

    return False


def is_in_comment(line: str, match_start: int) -> bool:
    """Check if the match position is inside a comment."""
    # Check for line comment before the match
    comment_pos = line.find('//')
    if comment_pos != -1 and comment_pos < match_start:
        return True
    return False


def check_file(filepath: Path) -> List[Tuple[int, str, str]]:
    """
    Check a single file for debug macros in production code.

    Returns list of (line_number, macro_name, line_content) tuples.
    """
    issues = []

    # Skip test files entirely
    filename = filepath.name
    if filename.startswith('test_') or filename.endswith('_test.rs'):
        return issues

    # Skip files in test directories
    path_str = str(filepath)
    if '/tests/' in path_str or '\\tests\\' in path_str:
        return issues
    if '/benches/' in path_str or '\\benches\\' in path_str:
        return issues

    try:
        content = filepath.read_text(encoding='utf-8')
    except (IOError, UnicodeDecodeError):
        return issues

    lines = content.split('\n')

    # Track if we're in a test module
    in_test_module = False
    brace_depth = 0
    test_module_start_depth = None

    for line_num, line in enumerate(lines):
        # Track brace depth for module scope
        brace_depth += line.count('{') - line.count('}')

        # Check for test module start
        if '#[cfg(test)]' in line or ('mod tests' in line and '{' in line):
            in_test_module = True
            test_module_start_depth = brace_depth - line.count('{')

        # Check for test module end
        if in_test_module and test_module_start_depth is not None:
            if brace_depth <= test_module_start_depth:
                in_test_module = False
                test_module_start_depth = None

        # Skip if in test context
        if in_test_module or is_test_context(line, lines, line_num):
            continue

        # Check for debug macros
        for pattern, macro_name in COMPILED_PATTERNS:
            for match in pattern.finditer(line):
                if not is_in_comment(line, match.start()):
                    issues.append((line_num + 1, macro_name, line.strip()))

    return issues


def main() -> int:
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: check_debug_macros.py <file1.rs> [file2.rs ...]", file=sys.stderr)
        return 0

    all_issues = []

    for filepath_str in sys.argv[1:]:
        filepath = Path(filepath_str)

        if not filepath.exists():
            continue

        if filepath.suffix != '.rs':
            continue

        issues = check_file(filepath)
        for line_num, macro_name, line_content in issues:
            all_issues.append((filepath, line_num, macro_name, line_content))

    if all_issues:
        print("Debug macros found in production code:", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        for filepath, line_num, macro_name, line_content in all_issues:
            print(f"{filepath}:{line_num}: {macro_name}", file=sys.stderr)
            print(f"    {line_content}", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print(f"\nFound {len(all_issues)} debug macro(s) in production code.", file=sys.stderr)
        print("Use proper error handling instead of todo!()/unimplemented!(),", file=sys.stderr)
        print("and remove dbg!() calls before committing.", file=sys.stderr)
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
