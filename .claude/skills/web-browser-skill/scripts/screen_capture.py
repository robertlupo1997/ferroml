#!/usr/bin/env python3
"""
Capture screenshots of web pages for visual verification.
Supports local dev servers and external URLs.
Token-optimized: outputs file path by default.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


def ensure_playwright():
    """Check if Playwright is available and provide install instructions."""
    if not PLAYWRIGHT_AVAILABLE:
        print("✗ Playwright not installed")
        print("  Install with: pip install playwright && playwright install chromium")
        sys.exit(1)


def generate_filename(prefix: str = "screenshot", output_dir: str = ".") -> str:
    """Generate timestamped filename."""
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return str(Path(output_dir) / f"{prefix}-{timestamp}.png")


def capture_screenshot(
    url: str = "http://localhost:5173",
    output: str = None,
    width: int = 1280,
    height: int = 720,
    full_page: bool = False,
    wait: int = 1000,
    selector: str = None
) -> dict:
    """Capture screenshot of a web page."""
    ensure_playwright()
    
    if output is None:
        output = generate_filename(output_dir="./screenshots")
    
    # Ensure output directory exists
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": width, "height": height}
        )
        page = context.new_page()
        
        try:
            page.goto(url, wait_until="networkidle", timeout=30000)
            page.wait_for_timeout(wait)  # Extra wait for animations
            
            if selector:
                # Capture specific element
                element = page.locator(selector)
                element.screenshot(path=output)
            else:
                # Capture full page or viewport
                page.screenshot(path=output, full_page=full_page)
            
            # Get page info
            title = page.title()
            
            return {
                "success": True,
                "path": output,
                "url": url,
                "title": title,
                "dimensions": f"{width}x{height}",
                "full_page": full_page
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "url": url
            }
        
        finally:
            browser.close()


def main():
    parser = argparse.ArgumentParser(description="Capture web page screenshots")
    parser.add_argument("url", nargs="?", default="http://localhost:5173", 
                        help="URL to capture (default: localhost:5173)")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--width", "-W", type=int, default=1280, help="Viewport width")
    parser.add_argument("--height", "-H", type=int, default=720, help="Viewport height")
    parser.add_argument("--full-page", "-f", action="store_true", help="Capture full page")
    parser.add_argument("--wait", "-w", type=int, default=1000, help="Wait time in ms after load")
    parser.add_argument("--selector", "-s", help="CSS selector to capture specific element")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()
    
    result = capture_screenshot(
        url=args.url,
        output=args.output,
        width=args.width,
        height=args.height,
        full_page=args.full_page,
        wait=args.wait,
        selector=args.selector
    )
    
    if args.json:
        print(json.dumps(result, indent=2))
    elif result["success"]:
        if args.verbose:
            print(f"✓ Screenshot saved: {result['path']}")
            print(f"  URL: {result['url']}")
            print(f"  Title: {result['title']}")
            print(f"  Size: {result['dimensions']}")
        else:
            print(f"✓ {result['path']}")
    else:
        print(f"✗ Failed: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
