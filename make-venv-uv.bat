@echo off

set CURRENT_DRIVE=%CD:~0,2%
set UV_CACHE_DIR=%CURRENT_DRIVE%\.uv-cache
echo Using UV_CACHE_DIR=%UV_CACHE_DIR%

setlocal enabledelayedexpansion

where uv >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [1/5] Installing uv...
    powershell -ExecutionPolicy Bypass -c "iwr https://astral.sh/uv/install.ps1 -useb | iex"
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
    echo uv installed successfully.
) else (
    echo DEBUG 1/5 Found uv
)

if exist ".venv" (
    echo DEBUG 2/5 Found .venv
) else (
    echo DEBUG 2/5 uv venv .venv
    uv venv .venv
)

set "EXPECTED_VENV_PATH=%CD%\.venv"
if /I "!VIRTUAL_ENV!" == "!EXPECTED_VENV_PATH!" (
    echo DEBUG 3/5 Found %VIRTUAL_ENV%
) else (
    echo DEBUG 3/5 call .venv\Scripts\activate
    call .venv\Scripts\activate
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to activate virtual environment!
        exit /b 1
    )
)

if not exist "requirements.txt" (
    echo DEBUG 5/5 Skipping lock file creation [no requirements.txt]
) else (
    if not exist "uv.lock" (
        echo DEBUG 5/5 uv pip compile requirements.txt -o uv.lock [new]
        uv pip compile requirements.txt -o uv.lock
        echo uv.lock created.
    ) else (
        for %%A in (requirements.txt) do set txt_time=%%~tA
        for %%B in (uv.lock) do set lock_time=%%~tB
        if "!txt_time!" GTR "!lock_time!" (
            echo DEBUG 5/5 uv pip compile requirements.txt -o uv.lock [renew]
            uv pip compile requirements.txt -o uv.lock
            echo uv.lock updated.
        ) else (
            echo DEBUG 5/5 uv.lock is up to date.
        )
    )
)

if exist uv.lock (
    echo DEBUG 4/5 uv pip sync uv.lock
    uv pip sync uv.lock
) else (
    if exist "requirements.txt" (
        echo DEBUG 4/5 uv pip sync requirements.txt
        uv pip sync requirements.txt
    ) else (
        echo DEBUG 4/5 Neither uv.lock nor requirements.txt
    )
)

endlocal & (
    REM Export varaibles updated by activate.bat to the parent shell
    set "VIRTUAL_ENV=%VIRTUAL_ENV%"
    set "PATH=%PATH%"
    set "PROMPT=%PROMPT%"
)