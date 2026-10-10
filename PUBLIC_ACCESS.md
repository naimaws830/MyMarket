# Exposing MyMarket MCP publicly

Makes the server reachable from the internet so a remote MCP client, or a
judge on another machine, can call it.

---

## Read this first

**There is no authentication on this server.** Anyone who learns the URL can
call every tool. That means they can:

- make your machine fetch RSS feeds and news article pages, repeatedly, from
  your IP address
- read your price data, your metrics and your demo cases
- write rows into your local SQLite database through the scan path

Nothing here can read files outside the project, run shell commands, or spend
money. The realistic worst case is someone hammering the endpoint until your
machine is busy or a news site rate limits your IP.

For a hackathon demo that is an acceptable trade. **Close the windows when you
are done.** These URLs die with the process, which is the main thing
protecting you.

---

## This network blocks Cloudflare tunnels

Measured on this machine:

| destination | port | result |
|---|---|---|
| region1.v2.argotunnel.com | 7844 TCP | **blocked** |
| region1/2.v2.argotunnel.com | 7844 UDP, QUIC | **blocked** |
| api.cloudflare.com | 443 | works |
| loca.lt | 443 | works |
| global.rel.tunnels.api.visualstudio.com | 443 | works |
| connect.ngrok-agent.com | 443 | works |

cloudflared's own pre-check calls this `Environment has critical failures`.
Every Cloudflare tunnel, quick or named, needs outbound 7844, and **no flag
works around it**, `--protocol http2` included, because that path uses 7844
too. That is why the quick tunnel printed a URL and then looped on
`failed to dial to edge`.

On this network, use a tunnel that runs over 443. On a phone hotspot
Cloudflare usually works again.

---

## The quick way

Double click **`serve_public.cmd`**. It uses localtunnel, which works here.
Pass `cloudflare` as an argument to use cloudflared instead, on a network that
allows 7844.

Two windows open: the server on port 8765, and the tunnel printing the public
URL. Your endpoint is that URL **with `/mcp` on the end**. Close both to stop.

---

## Option A, localtunnel. No account, works here

```
REM terminal 1, the server
cd C:\Users\UserPC\Desktop\MyMarket
set PYTHONPATH=%CD%\src
set PYTHONIOENCODING=utf-8
.venv\Scripts\python.exe -m mymarket.mcp_server --http --port 8765 --allow-host *

REM terminal 2, the tunnel
npx --yes localtunnel --port 8765
```

Wait for `serving on http://127.0.0.1:8765/mcp` before starting the tunnel.
The first start takes about 20 seconds while the embedding model loads.

localtunnel prints `your url is: https://something.loca.lt`.

**The one gotcha:** localtunnel serves a one time HTML reminder page. A caller
that gets HTML instead of JSON must send the header
`bypass-tunnel-reminder: true`.

Check it from any machine:

```bash
curl -s -X POST https://YOUR.loca.lt/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H 'bypass-tunnel-reminder: true' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'
```

Nine tool names means it works.

## Option B, VS Code port forwarding. Most reliable, needs a login

You already have VS Code and its tunnel CLI.

1. Start the server as in Option A, terminal 1.
2. Open the **Ports** panel, next to the Terminal tab.
3. **Forward a Port**, enter `8765`.
4. Right click the forwarded port, **Port Visibility**, set **Public**.
   Private means callers need your Microsoft or GitHub login.
5. Copy the `https://....devtunnels.ms` URL and append `/mcp`.

Runs over 443, nothing extra to install, and the URL is stable while the
window is open. Use this if localtunnel is flaky during the demo.

## Option C, ngrok. Needs a free account

`connect.ngrok-agent.com:443` is reachable, so ngrok would work. Sign up, get
an authtoken, then `ngrok http 8765`. Only worth it if A and B both fail.

---

## Why `--allow-host` is needed

The MCP SDK checks the `Host` header on every request and rejects anything it
does not recognise. That is correct protection against DNS rebinding, but a
tunnelled request arrives with `Host: something.loca.lt`, which is not
localhost, so every call would be refused for no obvious reason.

```
REM allow exactly one known hostname
--allow-host something.loca.lt

REM accept any host, what serve_public.cmd uses
--allow-host *
```

A tunnel hands out a new hostname each start, so `*` is the pragmatic choice
for a demo. The server logs a warning when you use it.

---

## Troubleshooting

| symptom | cause |
|---|---|
| `failed to dial to edge with quic` | Cloudflare only: port 7844 blocked. Use localtunnel |
| `error 1033` tunnel error | the server is not running, or is on a different port than the tunnel |
| HTML page instead of JSON | localtunnel reminder, send `bypass-tunnel-reminder: true` |
| `Invalid Host header` | `--allow-host` missing |
| `406 Not Acceptable` | `Accept` must list both `application/json` and `text/event-stream` |
| `[Errno 10048] ... bind` | port 8765 already in use by an earlier run, close that window |
| first call hangs about 20 seconds | normal, the embedding model is loading |

---

## For the demo tomorrow

**Use the Claude Desktop stdio connection, not a tunnel.** It is already
configured in `%APPDATA%\Claude\claude_desktop_config.json`, has one less
moving part, and does not depend on the venue network. The tunnel is for a
judge who wants to call it from their own machine, or for a teammate.

## If this needs to outlive the demo

A quick tunnel is anonymous, throwaway and unauthenticated. For anything
longer lived, put it behind Cloudflare Access or a VS Code private port, use a
named tunnel on your own domain for a stable URL, and add a shared secret
header check in front of the tool handlers. None of that is needed for a demo.
