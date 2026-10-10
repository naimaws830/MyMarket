@echo off
REM Serve MyMarket MCP over a public Cloudflare quick tunnel.
REM
REM WARNING: there is no authentication on this server. Anyone who learns the
REM URL can call every tool. Use it for a demo and close this window after.
REM
REM Two windows open: the server, and the tunnel that prints the public URL.

setlocal
set ROOT=%~dp0
set PYTHONPATH=%~dp0src
set PYTHONIOENCODING=utf-8
set PORT=8765

echo Starting MyMarket MCP server on port %PORT% ...
start "MyMarket MCP server" cmd /k ""%ROOT%.venv\Scripts\python.exe" -m mymarket.mcp_server --http --port %PORT% --allow-host *"

echo Waiting for the server to come up ...
timeout /t 12 /nobreak >nul

echo Opening the Cloudflare tunnel. The public URL appears below.
echo Your MCP endpoint is that URL with /mcp on the end.
echo.
"%ROOT%tools\cloudflared.exe" tunnel --url http://127.0.0.1:%PORT%
