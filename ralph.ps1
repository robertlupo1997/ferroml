# FerroML Ralph Loop (PowerShell)
# Usage: .\ralph.ps1 [-Mode plan|build]

param(
    [ValidateSet("plan", "build")]
    [string]$Mode = "build"
)

$ErrorActionPreference = "Stop"
$PromptFile = "PROMPT_$Mode.md"

if (-not (Test-Path $PromptFile)) {
    Write-Error "Error: $PromptFile not found"
    exit 1
}

Write-Host "=== FerroML Ralph Loop ===" -ForegroundColor Cyan
Write-Host "Mode: $Mode"
Write-Host "Press Ctrl+C to stop"
Write-Host ""

$iteration = 0
while ($true) {
    $iteration++
    Write-Host "--- Iteration $iteration ---" -ForegroundColor Yellow

    # Run Claude with the prompt
    Get-Content $PromptFile | claude --dangerously-skip-permissions

    # Run validation after each iteration
    Write-Host ""
    Write-Host "--- Validation ---" -ForegroundColor Green

    try {
        cargo check 2>&1
    } catch {
        Write-Host "cargo check failed" -ForegroundColor Red
    }

    try {
        cargo test 2>&1
    } catch {
        Write-Host "cargo test failed" -ForegroundColor Red
    }

    Write-Host ""
    Write-Host "--- Iteration $iteration complete ---" -ForegroundColor Yellow
    Write-Host ""

    # Small delay to allow Ctrl+C
    Start-Sleep -Seconds 2
}
