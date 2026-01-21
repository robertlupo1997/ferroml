#!/usr/bin/env python3
"""
Analyze page structure semantically - find elements by meaning, not coordinates.
Inspired by ios-simulator-skill's screen_mapper.py.
Token-optimized: 5-10 lines default output.
"""

import argparse
import json
import sys
from collections import Counter

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


def analyze_page(url: str = "http://localhost:5173", wait: int = 1000) -> dict:
    """Analyze page structure and interactive elements."""
    ensure_playwright()
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        try:
            page.goto(url, wait_until="networkidle", timeout=30000)
            page.wait_for_timeout(wait)
            
            # Get page metadata
            title = page.title()
            
            # Analyze elements using JavaScript
            analysis = page.evaluate("""() => {
                const result = {
                    buttons: [],
                    links: [],
                    inputs: [],
                    headings: [],
                    images: [],
                    forms: [],
                    interactive: [],
                    stats: {}
                };
                
                // Buttons
                document.querySelectorAll('button, [role="button"], input[type="submit"], input[type="button"]').forEach(el => {
                    const text = el.textContent?.trim() || el.value || el.getAttribute('aria-label') || '';
                    if (text) result.buttons.push(text.slice(0, 50));
                });
                
                // Links
                document.querySelectorAll('a[href]').forEach(el => {
                    const text = el.textContent?.trim() || el.getAttribute('aria-label') || '';
                    if (text && !text.startsWith('http')) result.links.push(text.slice(0, 50));
                });
                
                // Input fields
                document.querySelectorAll('input, textarea, select').forEach(el => {
                    const type = el.type || el.tagName.toLowerCase();
                    const label = el.placeholder || el.getAttribute('aria-label') || 
                                  document.querySelector(`label[for="${el.id}"]`)?.textContent?.trim() || '';
                    result.inputs.push({type, label: label.slice(0, 30)});
                });
                
                // Headings
                document.querySelectorAll('h1, h2, h3').forEach(el => {
                    const text = el.textContent?.trim();
                    if (text) result.headings.push({
                        level: el.tagName,
                        text: text.slice(0, 60)
                    });
                });
                
                // Images
                const images = document.querySelectorAll('img');
                result.stats.images = images.length;
                images.forEach(el => {
                    const alt = el.alt || el.getAttribute('aria-label') || '';
                    if (alt) result.images.push(alt.slice(0, 40));
                });
                
                // Forms
                document.querySelectorAll('form').forEach((form, i) => {
                    const inputs = form.querySelectorAll('input, textarea, select').length;
                    const submitBtn = form.querySelector('button[type="submit"], input[type="submit"]');
                    result.forms.push({
                        index: i,
                        inputs,
                        submitText: submitBtn?.textContent?.trim() || submitBtn?.value || 'Submit'
                    });
                });
                
                // Overall stats
                result.stats.totalElements = document.querySelectorAll('*').length;
                result.stats.interactive = document.querySelectorAll(
                    'button, a, input, textarea, select, [role="button"], [onclick]'
                ).length;
                result.stats.buttons = result.buttons.length;
                result.stats.links = result.links.length;
                result.stats.inputs = result.inputs.length;
                result.stats.forms = result.forms.length;
                
                return result;
            }""")
            
            return {
                "success": True,
                "url": url,
                "title": title,
                "analysis": analysis
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
    """Format analysis in compact, token-efficient style."""
    if not result["success"]:
        return f"✗ Failed to analyze: {result['error']}"
    
    a = result["analysis"]
    s = a["stats"]
    
    lines = [
        f"Page: {result['title'] or 'Untitled'} ({s['totalElements']} elements, {s['interactive']} interactive)"
    ]
    
    # Buttons
    if a["buttons"]:
        btns = ", ".join(f'"{b}"' for b in a["buttons"][:5])
        more = f" (+{len(a['buttons'])-5} more)" if len(a["buttons"]) > 5 else ""
        lines.append(f"Buttons: {btns}{more}")
    
    # Input fields
    if a["inputs"]:
        inputs_summary = []
        for inp in a["inputs"][:4]:
            label = inp["label"] or inp["type"]
            inputs_summary.append(label)
        lines.append(f"Inputs: {len(a['inputs'])} ({', '.join(inputs_summary)})")
    
    # Links (abbreviated)
    if a["links"]:
        lines.append(f"Links: {len(a['links'])} nav items")
    
    # Headings
    if a["headings"]:
        h1s = [h["text"] for h in a["headings"] if h["level"] == "H1"]
        if h1s:
            lines.append(f"Main heading: \"{h1s[0]}\"")
    
    # Forms
    if a["forms"]:
        for form in a["forms"]:
            lines.append(f"Form: {form['inputs']} fields → \"{form['submitText']}\"")
    
    return "\n".join(lines)


def format_verbose(result: dict) -> str:
    """Format analysis with full details."""
    if not result["success"]:
        return f"✗ Failed to analyze: {result['error']}"
    
    a = result["analysis"]
    s = a["stats"]
    
    lines = [
        f"═══ Page Analysis ═══",
        f"URL: {result['url']}",
        f"Title: {result['title'] or 'Untitled'}",
        f"Total elements: {s['totalElements']}",
        f"Interactive elements: {s['interactive']}",
        ""
    ]
    
    if a["headings"]:
        lines.append("─── Headings ───")
        for h in a["headings"]:
            lines.append(f"  {h['level']}: {h['text']}")
        lines.append("")
    
    if a["buttons"]:
        lines.append("─── Buttons ───")
        for btn in a["buttons"]:
            lines.append(f"  • {btn}")
        lines.append("")
    
    if a["inputs"]:
        lines.append("─── Input Fields ───")
        for inp in a["inputs"]:
            lines.append(f"  • [{inp['type']}] {inp['label'] or '(unlabeled)'}")
        lines.append("")
    
    if a["forms"]:
        lines.append("─── Forms ───")
        for form in a["forms"]:
            lines.append(f"  Form {form['index']}: {form['inputs']} inputs → \"{form['submitText']}\"")
        lines.append("")
    
    if a["links"]:
        lines.append(f"─── Links ({len(a['links'])}) ───")
        for link in a["links"][:10]:
            lines.append(f"  • {link}")
        if len(a["links"]) > 10:
            lines.append(f"  ... and {len(a['links'])-10} more")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze page structure semantically")
    parser.add_argument("url", nargs="?", default="http://localhost:5173",
                        help="URL to analyze (default: localhost:5173)")
    parser.add_argument("--wait", "-w", type=int, default=1000, help="Wait time after load (ms)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()
    
    result = analyze_page(url=args.url, wait=args.wait)
    
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
