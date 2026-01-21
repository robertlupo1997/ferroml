---
name: web-browser-skill
description: AI-driven frontend development with live browser preview and visual verification. Use when building web UIs (dashboards, landing pages, web apps), when previewing/testing frontend code, when researching web designs for inspiration, or when the user wants AI to "see" what it built. Works on Windows/Mac/Linux. Enables semantic navigation, screenshot verification, and design research.
---

# Web Browser Skill

Build, preview, and verify frontend code with real browser interaction. Inspired by ios-simulator-skill's semantic navigation philosophy, adapted for web browsers.

## Quick Start

```bash
# 1. Launch dev server + browser for current project
python scripts/dev_server.py

# 2. Capture what's on screen
python scripts/screen_capture.py --output screenshot.png

# 3. Analyze page structure
python scripts/page_analyzer.py

# 4. Research a design
python scripts/web_researcher.py --url "https://example.com" --capture
```

## Core Workflow

### Building Frontend

1. **Create project** from `assets/starter-template/` if none exists
2. **Write code** in React/TypeScript (or HTML/CSS if simpler)
3. **Launch** with `dev_server.py` to see live preview
4. **Capture** screenshot to verify visual output
5. **Iterate** based on user feedback

### Design Research

1. User provides inspiration URL or description
2. Use `web_researcher.py` to capture reference designs
3. Analyze layout, colors, typography
4. Adapt patterns to user's project

## Scripts Reference

| Script | Purpose | Default Output |
|--------|---------|----------------|
| `dev_server.py` | Launch Vite dev server + open browser | 1 line status |
| `screen_capture.py` | Screenshot current page | File path |
| `page_analyzer.py` | Semantic element analysis | 5-10 lines |
| `web_researcher.py` | Browse URL + capture | Screenshot + summary |
| `visual_diff.py` | Compare two screenshots | Diff highlights |

All scripts support `--verbose` for details, `--json` for parsing.

## Token Efficiency

Default output is minimal:
```
✓ Dev server running at http://localhost:5173
✓ Screenshot saved: ./screenshots/page-2024-01-15-143022.png
```

Use `--verbose` only when debugging.

## Working with Beginners

When user doesn't understand code:
1. Show screenshots of changes
2. Use plain language: "I moved the button to the right"
3. Offer correction phrases: "Say 'make it bigger' or 'change the color'"
4. Reference `references/MINIMUM_KNOWLEDGE.md` for teaching moments

## Playwright MCP Integration (Recommended)

**Prefer Playwright MCP** for direct browser control - it's faster and uses less context than writing scripts:

```bash
# Install once:
claude mcp add playwright -- npx @playwright/mcp@latest
```

**Playwright MCP handles:**
- Direct page interaction (clicks, typing, navigation)
- Form testing and validation
- Screenshots and visual verification
- Multi-page workflows

**Use this skill's scripts when:**
- MCP is unavailable or not configured
- You need dev server lifecycle management
- Complex custom automation logic is required

## Project Structure

For new projects, copy `assets/starter-template/`:
```
project/
├── src/
│   ├── App.tsx           # Main component
│   ├── components/       # Reusable pieces
│   └── main.tsx          # Entry point
├── index.html
├── package.json
├── vite.config.ts
└── tailwind.config.js
```

## See Also

- `references/MINIMUM_KNOWLEDGE.md` - What beginners need to correct AI
- `references/PROJECT_TEMPLATES.md` - Dashboard, Landing Page, Web App guides
- `references/COMMON_FIXES.md` - Frequent mistakes and solutions
