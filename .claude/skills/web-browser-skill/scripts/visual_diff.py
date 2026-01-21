#!/usr/bin/env python3
"""
Compare two screenshots to highlight visual differences.
Useful for verifying changes and catching regressions.
"""

import argparse
import json
import sys
from pathlib import Path

try:
    from PIL import Image, ImageChops, ImageDraw
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def ensure_pil():
    """Check if Pillow is available."""
    if not PIL_AVAILABLE:
        print("✗ Pillow not installed")
        print("  Install with: pip install pillow")
        sys.exit(1)


def compare_images(
    before_path: str,
    after_path: str,
    output_path: str = None,
    threshold: int = 30
) -> dict:
    """Compare two images and generate diff visualization."""
    ensure_pil()
    
    try:
        before = Image.open(before_path).convert("RGB")
        after = Image.open(after_path).convert("RGB")
    except Exception as e:
        return {"success": False, "error": f"Failed to open images: {e}"}
    
    # Check dimensions
    if before.size != after.size:
        return {
            "success": False,
            "error": f"Size mismatch: {before.size} vs {after.size}",
            "before_size": before.size,
            "after_size": after.size
        }
    
    # Calculate difference
    diff = ImageChops.difference(before, after)
    
    # Count different pixels
    diff_data = list(diff.getdata())
    total_pixels = len(diff_data)
    different_pixels = sum(
        1 for pixel in diff_data 
        if any(channel > threshold for channel in pixel)
    )
    
    diff_percentage = (different_pixels / total_pixels) * 100
    is_identical = different_pixels == 0
    
    # Generate output path if not provided
    if output_path is None:
        before_stem = Path(before_path).stem
        output_path = str(Path(before_path).parent / f"{before_stem}-diff.png")
    
    # Create highlighted diff image
    if not is_identical:
        # Enhance the diff for visibility
        highlighted = after.copy()
        draw = ImageDraw.Draw(highlighted)
        
        # Find regions with differences
        for y in range(0, after.height, 10):
            for x in range(0, after.width, 10):
                # Check 10x10 block
                block_diff = False
                for dy in range(min(10, after.height - y)):
                    for dx in range(min(10, after.width - x)):
                        idx = (y + dy) * after.width + (x + dx)
                        if idx < len(diff_data):
                            pixel = diff_data[idx]
                            if any(channel > threshold for channel in pixel):
                                block_diff = True
                                break
                    if block_diff:
                        break
                
                # Highlight changed blocks
                if block_diff:
                    draw.rectangle(
                        [x, y, min(x + 10, after.width), min(y + 10, after.height)],
                        outline="red",
                        width=2
                    )
        
        highlighted.save(output_path)
    else:
        output_path = None
    
    return {
        "success": True,
        "identical": is_identical,
        "different_pixels": different_pixels,
        "total_pixels": total_pixels,
        "diff_percentage": round(diff_percentage, 2),
        "diff_image": output_path,
        "dimensions": before.size
    }


def format_compact(result: dict) -> str:
    """Compact output."""
    if not result["success"]:
        return f"✗ {result['error']}"
    
    if result["identical"]:
        return "✓ Images are identical"
    
    return f"⚠ {result['diff_percentage']}% different ({result['different_pixels']:,} pixels) → {result['diff_image']}"


def format_verbose(result: dict) -> str:
    """Detailed output."""
    if not result["success"]:
        return f"✗ Comparison failed: {result['error']}"
    
    lines = ["═══ Visual Diff ═══"]
    
    if result["identical"]:
        lines.append("✓ Images are IDENTICAL")
    else:
        lines.extend([
            f"⚠ Images DIFFER",
            f"  Changed pixels: {result['different_pixels']:,} / {result['total_pixels']:,}",
            f"  Difference: {result['diff_percentage']}%",
            f"  Diff image: {result['diff_image']}"
        ])
    
    lines.append(f"  Dimensions: {result['dimensions'][0]}x{result['dimensions'][1]}")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare two screenshots")
    parser.add_argument("before", help="Path to before image")
    parser.add_argument("after", help="Path to after image")
    parser.add_argument("--output", "-o", help="Output path for diff image")
    parser.add_argument("--threshold", "-t", type=int, default=30,
                        help="Color difference threshold (0-255, default: 30)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()
    
    result = compare_images(
        before_path=args.before,
        after_path=args.after,
        output_path=args.output,
        threshold=args.threshold
    )
    
    if args.json:
        print(json.dumps(result, indent=2))
    elif args.verbose:
        print(format_verbose(result))
    else:
        print(format_compact(result))
    
    # Exit with code based on whether images differ
    if not result["success"]:
        sys.exit(2)
    elif not result["identical"]:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
