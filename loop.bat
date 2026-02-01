@echo off
REM FerroML Ralph Wiggum Loop for Windows
REM Usage:
REM   loop.bat           - Build mode, unlimited iterations
REM   loop.bat plan      - Plan mode, unlimited iterations
REM   loop.bat plan 5    - Plan mode, max 5 iterations
REM   loop.bat 20        - Build mode, max 20 iterations

setlocal enabledelayedexpansion

set MODE=build
set MAX_ITERATIONS=0
set ITERATION=0

REM Parse arguments
for %%a in (%*) do (
    if "%%a"=="plan" (
        set MODE=plan
    ) else (
        set /a "test=%%a" 2>nul
        if !errorlevel!==0 set MAX_ITERATIONS=%%a
    )
)

set PROMPT_FILE=PROMPT_%MODE%.md

if not exist "%PROMPT_FILE%" (
    echo Error: %PROMPT_FILE% not found
    exit /b 1
)

echo ==============================================
echo FerroML Ralph Loop
echo Mode: %MODE%
if %MAX_ITERATIONS%==0 (echo Max iterations: unlimited) else (echo Max iterations: %MAX_ITERATIONS%)
echo Prompt file: %PROMPT_FILE%
echo ==============================================
echo.

:loop
set /a ITERATION+=1

echo.
echo ==============================================
echo ITERATION %ITERATION% - %date% %time%
echo ==============================================
echo.

REM Run Claude with the prompt
type "%PROMPT_FILE%" | claude -p --dangerously-skip-permissions --model claude-sonnet-4-20250514 --verbose

if errorlevel 1 (
    echo.
    echo Claude exited with error. Waiting 30 seconds before retry...
    timeout /t 30 /nobreak >nul
)

REM Check iteration limit
if not %MAX_ITERATIONS%==0 (
    if %ITERATION% geq %MAX_ITERATIONS% (
        echo.
        echo ==============================================
        echo Reached max iterations ^(%MAX_ITERATIONS%^)
        echo ==============================================
        goto :end
    )
)

echo.
echo Iteration %ITERATION% complete. Starting next iteration in 5 seconds...
timeout /t 5 /nobreak >nul
goto :loop

:end
echo.
echo Ralph loop finished after %ITERATION% iterations
