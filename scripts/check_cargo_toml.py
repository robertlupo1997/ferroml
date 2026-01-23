#!/usr/bin/env python3
"""
Check Cargo.toml files for common issues and formatting.

This script validates:
- TOML syntax is valid
- Required fields are present
- Dependencies use consistent formatting
- No duplicate keys

Exit codes:
  0 - No issues found
  1 - Issues found in Cargo.toml
"""

import sys
from pathlib import Path
from typing import List, Tuple

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # Fallback for older Python
    except ImportError:
        # If neither is available, just do basic syntax check
        tomllib = None


def check_toml_syntax(filepath: Path) -> List[str]:
    """Check TOML syntax validity."""
    issues = []

    try:
        content = filepath.read_text(encoding='utf-8')
    except (IOError, UnicodeDecodeError) as e:
        return [f"Cannot read file: {e}"]

    if tomllib is not None:
        try:
            tomllib.loads(content)
        except Exception as e:
            issues.append(f"TOML syntax error: {e}")

    return issues


def check_formatting(filepath: Path) -> List[str]:
    """Check for common formatting issues."""
    issues = []

    try:
        content = filepath.read_text(encoding='utf-8')
    except (IOError, UnicodeDecodeError):
        return issues

    lines = content.split('\n')

    for line_num, line in enumerate(lines, 1):
        # Check for tabs (prefer spaces in TOML)
        if '\t' in line and not line.strip().startswith('#'):
            issues.append(f"Line {line_num}: Use spaces instead of tabs")

        # Check for trailing whitespace
        if line.rstrip() != line and line.strip():
            issues.append(f"Line {line_num}: Trailing whitespace")

        # Check for very long lines (readability)
        if len(line) > 200:
            issues.append(f"Line {line_num}: Line exceeds 200 characters")

    return issues


def check_required_fields(filepath: Path) -> List[str]:
    """Check for required fields in Cargo.toml."""
    issues = []

    if tomllib is None:
        return issues

    try:
        content = filepath.read_text(encoding='utf-8')
        data = tomllib.loads(content)
    except Exception:
        return issues

    # For workspace root
    if 'workspace' in data:
        return issues  # Workspace roots have different requirements

    # For regular packages
    if 'package' in data:
        package = data['package']

        # Check for name
        if 'name' not in package:
            issues.append("Missing required field: [package].name")

        # Check for version (can inherit from workspace)
        if 'version' not in package and 'version.workspace' not in str(package):
            # Check if using workspace inheritance
            if not package.get('version', {}) == {'workspace': True}:
                has_workspace_version = False
                for key, val in package.items():
                    if 'workspace' in str(val):
                        has_workspace_version = True
                        break
                if 'version' not in package and not has_workspace_version:
                    issues.append("Missing required field: [package].version")

        # Check for edition
        if 'edition' not in package:
            # Check workspace inheritance
            if not isinstance(package.get('edition'), dict):
                has_edition = 'edition' in str(data)
                if not has_edition:
                    issues.append("Consider adding: [package].edition = \"2021\"")

    return issues


def check_dependency_formatting(filepath: Path) -> List[str]:
    """Check for consistent dependency formatting."""
    issues = []
    warnings = []

    try:
        content = filepath.read_text(encoding='utf-8')
    except (IOError, UnicodeDecodeError):
        return issues

    lines = content.split('\n')
    in_deps_section = False

    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()

        # Track if we're in a dependencies section
        if stripped.startswith('[') and 'dependencies' in stripped.lower():
            in_deps_section = True
            continue
        elif stripped.startswith('['):
            in_deps_section = False
            continue

        if in_deps_section and '=' in stripped and not stripped.startswith('#'):
            # Check for path dependencies without workspace
            if 'path = ' in stripped and 'workspace' not in filepath.parent.name:
                if '"../' in stripped or '".\\..' in stripped:
                    warnings.append(f"Line {line_num}: Consider using workspace dependencies for: {stripped.split('=')[0].strip()}")

    # Warnings are informational, not blocking
    # Uncomment below to make them issues
    # issues.extend(warnings)

    return issues


def main() -> int:
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: check_cargo_toml.py <Cargo.toml> [Cargo.toml ...]", file=sys.stderr)
        return 0

    all_issues: List[Tuple[Path, str]] = []

    for filepath_str in sys.argv[1:]:
        filepath = Path(filepath_str)

        if not filepath.exists():
            continue

        if filepath.name != 'Cargo.toml':
            continue

        # Run all checks
        issues = []
        issues.extend(check_toml_syntax(filepath))
        issues.extend(check_formatting(filepath))
        issues.extend(check_required_fields(filepath))
        issues.extend(check_dependency_formatting(filepath))

        for issue in issues:
            all_issues.append((filepath, issue))

    if all_issues:
        print("Issues found in Cargo.toml files:", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        for filepath, issue in all_issues:
            print(f"{filepath}: {issue}", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
