@echo off
REM Serve MyMarket MCP publicly.
REM
REM   serve_public.cmd                      random localtunnel URL
REM   serve_public.cmd mymarket-my          STABLE URL: https://mymarket-my.loca.lt
REM   serve_public.cmd ngrok                ngrok, uses your reserved domain
REM   serve_public.cmd cloudflare           cloudflared, needs outbound port 7844
REM
REM A requested localtunnel subdomain is first come first served, not
REM reserved. If someone else holds it you get a random one instead, so check
REM the printed URL before you publish it anywhere.
REM
REM WARNING: no authentication. Anyone with the URL can call every tool.

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
if /i "%~1"=="ngrok" goto ngrok

if "%~1"=="" (
  echo Opening a localtunnel with a random URL.
  echo Tip: pass a name to request a stable one, e.g. serve_public.cmd mymarket-my
  echo.
  npx --yes localtunnel --port %PORT%
) else (
  echo Requesting the stable subdomain "%~1" ...
  echo Your endpoint will be https://%~1.loca.lt/mcp if it is free.
  echo.
  npx --yes localtunnel --port %PORT% --subdomain %~1
)
goto :eof

:ngrok
echo Starting ngrok. Set your reserved domain in the command below first.
echo   ngrok http %PORT% --domain=YOUR-NAME.ngrok-free.app
echo.
ngrok http %PORT%
goto :eof

:cloudflare
echo Opening a Cloudflare tunnel. Needs outbound port 7844, which this
echo network blocks. Expect "QUIC connection failed" here.
echo.
"%ROOT%tools\cloudflared.exe" tunnel --url http://127.0.0.1:%PORT%
