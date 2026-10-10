@echo off
REM Serve MyMarket MCP publicly.
REM
REM Usage:  serve_public.cmd            uses localtunnel  (port 443, works on
REM                                     networks that block Cloudflare)
REM         serve_public.cmd cloudflare uses cloudflared   (needs port 7844)
REM
REM WARNING: no authentication. Anyone with the URL can call every tool.
REM Close both windows when you are done.

setlocal
set ROOT=%~dp0
set PYTHONPATH=%~dp0src
set PYTHONIOENCODING=utf-8
set PORT=8765

echo Starting MyMarket MCP server on port %PORT% ...
start "MyMarket MCP server" cmd /k ""%ROOT%.venv\Scripts\python.exe" -m mymarket.mcp_server --http --port %PORT% --allow-host *"

echo Waiting for the server to load the embedding model ...
timeout /t 20 /nobreak >nul

if /i "%~1"=="cloudflare" goto cloudflare

echo.
echo Opening a localtunnel. Your MCP endpoint is the printed URL plus /mcp
echo If a caller gets an HTML reminder page instead of JSON, they must send
echo the header:  bypass-tunnel-reminder: true
echo.
npx --yes localtunnel --port %PORT%
goto :eof

:cloudflare
echo.
echo Opening a Cloudflare tunnel. This needs outbound port 7844.
echo If you see "QUIC connection failed" your network blocks it: press Ctrl+C
echo and run serve_public.cmd with no argument to use localtunnel instead.
echo.
"%ROOT%tools\cloudflared.exe" tunnel --url http://127.0.0.1:%PORT%
