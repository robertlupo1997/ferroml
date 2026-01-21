#!/usr/bin/env python3
"""
Launch development server and open browser for live preview.
Token-optimized output: 1 line default, verbose for debugging.
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


def find_project_root() -> Path:
    """Find project root by looking for package.json or index.html."""
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / "package.json").exists():
            return parent
        if (parent / "index.html").exists():
            return parent
    return current


def get_package_manager() -> str:
    """Detect available package manager."""
    if shutil.which("pnpm"):
        return "pnpm"
    if shutil.which("yarn"):
        return "yarn"
    return "npm"


def check_dependencies(project_root: Path) -> bool:
    """Check if node_modules exists."""
    return (project_root / "node_modules").exists()


def install_dependencies(project_root: Path, verbose: bool = False) -> bool:
    """Install npm dependencies."""
    pm = get_package_manager()
    cmd = [pm, "install"]
    
    if verbose:
        print(f"Installing dependencies with {pm}...")
    
    result = subprocess.run(
        cmd,
        cwd=project_root,
        capture_output=not verbose,
        text=True
    )
    return result.returncode == 0


def find_dev_script(project_root: Path) -> str:
    """Find the dev script name from package.json."""
    package_json = project_root / "package.json"
    if package_json.exists():
        with open(package_json) as f:
            data = json.load(f)
            scripts = data.get("scripts", {})
            for name in ["dev", "start", "serve"]:
                if name in scripts:
                    return name
    return "dev"


def launch_server(project_root: Path, port: int = 5173, verbose: bool = False) -> subprocess.Popen:
    """Launch the development server."""
    pm = get_package_manager()
    script = find_dev_script(project_root)
    
    # Build command with port override
    if pm == "npm":
        cmd = [pm, "run", script, "--", "--port", str(port)]
    else:
        cmd = [pm, script, "--port", str(port)]
    
    if verbose:
        print(f"Running: {' '.join(cmd)}")
    
    # Launch server process
    process = subprocess.Popen(
        cmd,
        cwd=project_root,
        stdout=subprocess.PIPE if not verbose else None,
        stderr=subprocess.PIPE if not verbose else None,
        text=True
    )
    
    return process


def wait_for_server(url: str, timeout: int = 30) -> bool:
    """Wait for server to be ready."""
    import urllib.request
    import urllib.error
    
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except (urllib.error.URLError, ConnectionRefusedError):
            time.sleep(0.5)
    return False


def main():
    parser = argparse.ArgumentParser(description="Launch dev server and open browser")
    parser.add_argument("--port", type=int, default=5173, help="Port number (default: 5173)")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--install", action="store_true", help="Force dependency install")
    args = parser.parse_args()
    
    project_root = find_project_root()
    url = f"http://localhost:{args.port}"
    
    # Check for package.json
    if not (project_root / "package.json").exists():
        error = "No package.json found. Create project first or copy from assets/starter-template/"
        if args.json:
            print(json.dumps({"success": False, "error": error}))
        else:
            print(f"✗ {error}")
        sys.exit(1)
    
    # Install dependencies if needed
    if args.install or not check_dependencies(project_root):
        if args.verbose:
            print("Installing dependencies...")
        if not install_dependencies(project_root, args.verbose):
            error = "Failed to install dependencies"
            if args.json:
                print(json.dumps({"success": False, "error": error}))
            else:
                print(f"✗ {error}")
            sys.exit(1)
    
    # Launch server
    process = launch_server(project_root, args.port, args.verbose)
    
    # Wait for server to be ready
    if args.verbose:
        print(f"Waiting for server at {url}...")
    
    if wait_for_server(url):
        # Open browser
        if not args.no_browser:
            webbrowser.open(url)
        
        if args.json:
            print(json.dumps({
                "success": True,
                "url": url,
                "port": args.port,
                "project": str(project_root),
                "pid": process.pid
            }))
        elif args.verbose:
            print(f"✓ Dev server running at {url}")
            print(f"  Project: {project_root}")
            print(f"  PID: {process.pid}")
            print("  Press Ctrl+C to stop")
        else:
            print(f"✓ Dev server running at {url}")
        
        # Keep running until interrupted
        try:
            process.wait()
        except KeyboardInterrupt:
            process.terminate()
            print("\n✓ Server stopped")
    else:
        process.terminate()
        error = f"Server failed to start on port {args.port}"
        if args.json:
            print(json.dumps({"success": False, "error": error}))
        else:
            print(f"✗ {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
