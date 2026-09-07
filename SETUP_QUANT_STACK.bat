@echo off
setlocal
echo.
echo ================================================
echo   Quant-Stack - One-time setup + first pull
echo ================================================
echo.

cd /d "%~dp0"
set "ROOT=%CD%"
cd /d "%ROOT%\quant-stack"
echo Working folder: %CD%
echo.

REM ---------------------------------------------------------------
REM 1. uv (installs Python for you if needed)
REM ---------------------------------------------------------------
where uv >nul 2>nul
if errorlevel 1 (
    echo uv is not installed. Installing it now...
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
    where uv >nul 2>nul
    if errorlevel 1 (
        echo.
        echo ERROR: uv still not found. Close this window, open a NEW one,
        echo and run this file again ^(the installer updates PATH for new windows^).
        pause
        exit /b 1
    )
)
echo uv found.
echo.

echo Installing the toolbox (this can take a few minutes the first time)...
uv sync --all-extras
if errorlevel 1 (
    echo.
    echo ERROR: uv sync failed - see the message above.
    pause
    exit /b 1
)
echo.

REM ---------------------------------------------------------------
REM 2. Schwab app key + secret -> keychain, from a .env you already have
REM ---------------------------------------------------------------
set "ENVFILE="
if exist "%ROOT%\quant-stack\terminal\.env" set "ENVFILE=%ROOT%\quant-stack\terminal\.env"
if not defined ENVFILE if exist "%ROOT%\.env" set "ENVFILE=%ROOT%\.env"
if defined ENVFILE (
    echo Found Schwab keys in: %ENVFILE%
    uv run python -m config.secrets --set-schwab-app-from-env-file "%ENVFILE%"
) else (
    echo No .env with SCHWAB_API_KEY found. You will be asked to paste them:
    uv run python -m config.secrets --set-schwab-app
)
if errorlevel 1 (
    echo.
    echo ERROR: could not store the app key/secret - see above.
    pause
    exit /b 1
)
echo.

REM ---------------------------------------------------------------
REM 3. Schwab token -> keychain, from a token file you already have
REM ---------------------------------------------------------------
uv run python -c "import sys; from config.secrets import get_schwab_token; sys.exit(0 if get_schwab_token(env='production') else 1)" >nul 2>nul
if not errorlevel 1 (
    echo A Schwab token is already in the keychain - skipping token import.
    goto :token_done
)
set "TOKENFILE="
for %%F in ("%ROOT%\schwab_token.json" "%ROOT%\quant-stack\terminal\schwab_token.json" "%ROOT%\quant-stack\schwab_token.json") do (
    if not defined TOKENFILE if exist "%%~F" set "TOKENFILE=%%~F"
)
if defined TOKENFILE (
    echo Found token file: %TOKENFILE%
    uv run python -m config.secrets --import-token-file "%TOKENFILE%"
) else (
    echo No schwab_token.json found. Starting the one-time Schwab login...
    uv run python -m cli connect schwab
)
if errorlevel 1 (
    echo.
    echo ERROR: could not store the Schwab token - see above.
    pause
    exit /b 1
)
:token_done
echo.

REM ---------------------------------------------------------------
REM 4. Turn the broker on
REM ---------------------------------------------------------------
if not exist ".env" type nul > .env
findstr /C:"BROKERS_ENABLED" .env >nul 2>nul
if errorlevel 1 (
    echo BROKERS_ENABLED={"schwab": true}>> .env
    echo Enabled schwab in quant-stack\.env
) else (
    echo BROKERS_ENABLED already set in quant-stack\.env
)
echo.

REM ---------------------------------------------------------------
REM 5. The moment of truth: one real chain from Schwab
REM ---------------------------------------------------------------
echo ================================================
echo   Pulling today's QQQ option chain from Schwab...
echo ================================================
echo.
uv run python -m cli capture-chains QQQ
echo.
if errorlevel 1 (
    echo RESULT: FAILED - copy everything above and send it to Claude.
) else (
    echo RESULT: SUCCESS - the archive has its first real chain.
    echo Next: schedule this command to run every weekday at 4:35 PM:
    echo     uv run python -m cli capture-chains QQQ SPY
)
echo.
pause
endlocal
