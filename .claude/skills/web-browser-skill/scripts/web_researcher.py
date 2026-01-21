#!/usr/bin/env python3
"""
Research web designs for inspiration.
Browse URLs, capture screenshots, analyze design patterns.
Token-optimized: screenshot path + brief summary by default.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


def ensure_playwright():
    """Check if Playwright is available."""
    if not PLAYWRIGHT_AVAILABLE:
        print("✗ Playwright not installed")
        print("  Install with: pip install playwright && playwright install chromium")
        sys.exit(1)


def sanitize_filename(url: str) -> str:
    """Create safe filename from URL."""
    parsed = urlparse(url)
    name = parsed.netloc.replace(".", "-")
    if parsed.path and parsed.path != "/":
        name += parsed.path.replace("/", "-")[:30]
    return name


def research_url(
    url: str,
    output_dir: str = "./research",
    width: int = 1440,
    height: int = 900,
    full_page: bool = True,
    analyze: bool = True
) -> dict:
    """Capture and analyze a web page for design research."""
    ensure_playwright()
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{sanitize_filename(url)}-{timestamp}.png"
    screenshot_path = str(Path(output_dir) / filename)
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": width, "height": height})
        page = context.new_page()
        
        try:
            page.goto(url, wait_until="networkidle", timeout=30000)
            page.wait_for_timeout(2000)  # Wait for animations
            
            # Capture screenshot
            page.screenshot(path=screenshot_path, full_page=full_page)
            
            # Get page metadata
            title = page.title()
            
            # Analyze design patterns
            design_analysis = {}
            if analyze:
                design_analysis = page.evaluate("""() => {
                    const result = {
                        colors: new Set(),
                        fonts: new Set(),
                        layout: {},
                        components: []
                    };
                    
                    // Sample colors from key elements
                    const elements = document.querySelectorAll('body, header, nav, main, footer, button, a, h1, h2');
                    elements.forEach(el => {
                        const style = getComputedStyle(el);
                        if (style.backgroundColor !== 'rgba(0, 0, 0, 0)') {
                            result.colors.add(style.backgroundColor);
                        }
                        result.colors.add(style.color);
                        result.fonts.add(style.fontFamily.split(',')[0].trim().replace(/"/g, ''));
                    });
                    
                    // Detect layout type
                    const body = document.body;
                    const main = document.querySelector('main, [role="main"], .main, #main');
                    if (main) {
                        const style = getComputedStyle(main);
                        result.layout.mainWidth = style.maxWidth || 'full';
                        result.layout.display = style.display;
                    }
                    
                    // Check for common components
                    if (document.querySelector('nav, [role="navigation"]')) result.components.push('navigation');
                    if (document.querySelector('header')) result.components.push('header');
                    if (document.querySelector('footer')) result.components.push('footer');
                    if (document.querySelector('.hero, [class*="hero"]')) result.components.push('hero section');
                    if (document.querySelector('.card, [class*="card"]')) result.components.push('cards');
                    if (document.querySelector('.grid, [class*="grid"]')) result.components.push('grid layout');
                    if (document.querySelector('.sidebar, aside')) result.components.push('sidebar');
                    if (document.querySelector('form')) result.components.push('form');
                    if (document.querySelector('.modal, [role="dialog"]')) result.components.push('modal');
                    
                    // Check for dark/light theme
                    const bgColor = getComputedStyle(body).backgroundColor;
                    const rgb = bgColor.match(/\\d+/g);
                    if (rgb) {
                        const brightness = (parseInt(rgb[0]) + parseInt(rgb[1]) + parseInt(rgb[2])) / 3;
                        result.layout.theme = brightness < 128 ? 'dark' : 'light';
                    }
                    
                    return {
                        colors: [...result.colors].slice(0, 8),
                        fonts: [...result.fonts].slice(0, 4),
                        layout: result.layout,
                        components: result.components
                    };
                }""")
            
            return {
                "success": True,
                "url": url,
                "title": title,
                "screenshot": screenshot_path,
                "design": design_analysis
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "url": url
            }
        
        finally:
            browser.close()


def format_compact(result: dict) -> str:
    """Compact output for token efficiency."""
    if not result["success"]:
        return f"✗ Research failed: {result['error']}"
    
    d = result.get("design", {})
    lines = [
        f"✓ {result['screenshot']}",
        f"  {result['title']}"
    ]
    
    if d.get("layout", {}).get("theme"):
        lines.append(f"  Theme: {d['layout']['theme']}")
    
    if d.get("components"):
        lines.append(f"  Components: {', '.join(d['components'][:5])}")
    
    if d.get("fonts"):
        lines.append(f"  Fonts: {', '.join(d['fonts'][:3])}")
    
    return "\n".join(lines)


def format_verbose(result: dict) -> str:
    """Detailed design analysis."""
    if not result["success"]:
        return f"✗ Research failed: {result['error']}"
    
    d = result.get("design", {})
    
    lines = [
        f"═══ Design Research ═══",
        f"URL: {result['url']}",
        f"Title: {result['title']}",
        f"Screenshot: {result['screenshot']}",
        ""
    ]
    
    if d.get("layout"):
        lines.append("─── Layout ───")
        for k, v in d["layout"].items():
            lines.append(f"  {k}: {v}")
        lines.append("")
    
    if d.get("components"):
        lines.append("─── Components Detected ───")
        for comp in d["components"]:
            lines.append(f"  • {comp}")
        lines.append("")
    
    if d.get("fonts"):
        lines.append("─── Typography ───")
        for font in d["fonts"]:
            lines.append(f"  • {font}")
        lines.append("")
    
    if d.get("colors"):
        lines.append("─── Color Palette ───")
        for color in d["colors"][:6]:
            lines.append(f"  • {color}")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Research web designs for inspiration")
    parser.add_argument("url", help="URL to research")
    parser.add_argument("--output", "-o", default="./research", help="Output directory")
    parser.add_argument("--width", "-W", type=int, default=1440, help="Viewport width")
    parser.add_argument("--height", "-H", type=int, default=900, help="Viewport height")
    parser.add_argument("--viewport-only", action="store_true", help="Capture viewport only (not full page)")
    parser.add_argument("--no-analyze", action="store_true", help="Skip design analysis")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()
    
    result = research_url(
        url=args.url,
        output_dir=args.output,
        width=args.width,
        height=args.height,
        full_page=not args.viewport_only,
        analyze=not args.no_analyze
    )
    
    if args.json:
        print(json.dumps(result, indent=2))
    elif args.verbose:
        print(format_verbose(result))
    else:
        print(format_compact(result))
    
    if not result["success"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
