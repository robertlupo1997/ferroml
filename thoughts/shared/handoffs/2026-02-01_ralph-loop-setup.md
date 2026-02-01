---
date: 2026-02-01T02:10:00-05:00
researcher: Claude
git_commit: 1510810
git_branch: master
repository: ferroml
topic: Ralph Wiggum Loop Setup Research
tags: [ralph-loop, automation, ci-cd, autonomous-development]
status: needs-research
---

# Handoff: Ralph Loop Setup Research Needed

## Context

We've set up the basic Ralph Wiggum loop files for FerroML:
- `loop.sh` - Bash script for Unix/WSL
- `loop.bat` - Windows batch file
- `PROMPT_plan.md` - Planning mode prompt
- `PROMPT_build.md` - Build mode prompt
- `AGENTS.md` - Project-specific operational reference
- `IMPLEMENTATION_PLAN.md` - Task tracking with 17 testing phases

## Problem

The loop isn't running reliably on Windows. Issues encountered:
1. Git Bash has cygpath conflicts when running the loop
2. Windows batch files don't run easily from PowerShell (`.\loop.bat` required)
3. WSL cron only runs when WSL terminal is open
4. Need a way to run overnight autonomously

## Research Task

**Research how people properly set up Ralph Wiggum loops for autonomous overnight operation.**

### Questions to Answer

1. **Execution Environment**
   - Do people run Ralph loops on local machines or cloud VMs?
   - What's the recommended OS (Linux VM, WSL, Docker)?
   - How do they handle Windows environments?

2. **Scheduling**
   - Windows Task Scheduler vs WSL cron vs other tools?
   - How to keep WSL alive for cron jobs?
   - Cloud scheduling options (GitHub Actions, etc.)?

3. **Monitoring**
   - How do people monitor overnight Ralph runs?
   - Logging best practices?
   - Alerting on failures?

4. **Security**
   - The `--dangerously-skip-permissions` flag is required
   - How do people isolate Ralph runs for safety?
   - Docker containers? Fly.io? E2B?

5. **Cost Management**
   - How to limit API costs for overnight runs?
   - Iteration limits? Time limits?

### Resources to Check

1. https://github.com/ghuntley/how-to-ralph-wiggum - Original guide
2. Search for "Ralph Wiggum Claude loop" implementations
3. Look for GitHub repos with working ralph loop setups
4. Check Claude Code documentation for headless operation

## Current Files

```
ferroml/
├── loop.sh              # Bash version (works in WSL/Linux)
├── loop.bat             # Windows batch (needs testing)
├── setup_ralph.sh       # WSL setup helper
├── PROMPT_plan.md       # Planning mode
├── PROMPT_build.md      # Build mode
├── AGENTS.md            # Project reference
└── IMPLEMENTATION_PLAN.md # Task tracking
```

## What Needs to Happen

1. Research proper Ralph loop setups
2. Determine best execution environment for Windows user
3. Set up reliable overnight scheduling
4. Test the loop runs successfully
5. Monitor first overnight run

## Quick Test Commands

```bash
# In WSL:
cd /mnt/c/Users/Trey/Downloads/ferroml
./loop.sh plan 1  # Single planning iteration

# Check if claude is installed in WSL:
which claude
npm install -g @anthropic-ai/claude-code  # If not installed
```

## Success Criteria

- [ ] Ralph loop runs unattended overnight
- [ ] Tasks get completed and committed
- [ ] Failures are logged and don't crash the loop
- [ ] Can check progress in the morning via git log
