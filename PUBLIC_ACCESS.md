# Exposing MyMarket MCP over a Cloudflare tunnel

Makes the server reachable from the internet so a remote MCP client, or a
judge on another machine, can call it.

---

## Read this first

**There is no authentication on this server.** Anyone who learns the URL can
call every tool. That means they can:

- make your machine fetch RSS feeds and news article pages, repeatedly, from
  your IP address
- read your price data, your metrics and your demo cases
- write rows into your local SQLite database through the daily scan path

Nothing here can read files outside the project, run shell commands, or spend
money. The realistic worst case is someone hammering the endpoint until your
machine is busy or a news site rate limits your IP.

For a hackathon demo that is an acceptable trade. **Close the tunnel window
when you are done.** A quick tunnel URL dies with the process, which is the
main thing protecting you.

---

## The quick way

Double click **`serve_public.cmd`** in the project root.

It opens two windows:

1. the MCP server on port 8765
2. cloudflared, which prints the public URL

Your endpoint is that URL **with `/mcp` on the end**:

```
https://something-random-words.trycloudflare.com/mcp
```

To stop, close both windows.

---

## The manual way, so you know what it does

**Terminal 1, the server:**

```cmd
cd C:\Users\UserPC\Desktop\MyMarket
set PYTHONPATH=%CD%\src
set PYTHONIOENCODING=utf-8
.venv\Scripts\python.exe -m mymarket.mcp_server --http --port 8765 --allow-host *
```

Wait for `serving on http://127.0.0.1:8765/mcp`. First start takes about 20
seconds because it loads the embedding model.

**Terminal 2, the tunnel:**

```cmd
cd C:\Users\UserPC\Desktop\MyMarket
tools\cloudflared.exe tunnel --url http://127.0.0.1:8765
```

cloudflared prints a box containing the public URL. Append `/mcp`.

---

## Why `--allow-host` is needed

The MCP SDK checks the `Host` header on every request and rejects anything it
does not recognise. That is correct protection against DNS rebinding, but a
tunnelled request arrives with `Host: something.trycloudflare.com`, which is
not localhost, so every call would be refused with no obvious reason.

Options, least permissive first:

```cmd
REM once you know the tunnel hostname, allow exactly that one
--allow-host something-random-words.trycloudflare.com

REM accept any host. Simplest, and what serve_public.cmd uses.
--allow-host *
```

A quick tunnel gives you a new random hostname every start, so pinning the
exact host means editing the command each time. `*` is the pragmatic choice
for a demo, and the server logs a warning when you use it.

---

## Check it works before you rely on it

From any machine:

```bash
curl -s -X POST https://YOUR-URL.trycloudflare.com/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'
```

You should see the nine tool names. If instead you get:

| symptom | cause |
|---|---|
| `error 1033`, tunnel error | the server is not running, or is on a different port than the tunnel |
| `Invalid Host header` | the `--allow-host` flag is missing |
| `406 Not Acceptable` | the `Accept` header must list both `application/json` and `text/event-stream` |
| hangs about 20 seconds then works | normal on the first call, the embedding model is loading |

---

## Connecting a client to the public URL

**Claude Desktop** connects to local servers over stdio, which is what
`%APPDATA%\Claude\claude_desktop_config.json` is already set up for. That
config does not need the tunnel and is the more reliable path for a live demo.
Use the tunnel for remote clients, for a teammate's machine, or for anything
that speaks MCP over HTTP.

**A generic HTTP MCP client** points at `https://YOUR-URL.trycloudflare.com/mcp`
using the streamable HTTP transport.

The server runs stateless in HTTP mode, so every request is independent and
there is no session to keep alive across a flaky connection.

---

## If you want this to survive past the demo

A quick tunnel is anonymous, throwaway and unauthenticated. For anything
longer lived:

1. Put the endpoint behind **Cloudflare Access** so callers must authenticate.
2. Use a **named tunnel** bound to your own domain, so the URL is stable:
   `cloudflared tunnel login`, then `cloudflared tunnel create mymarket`.
3. Add a shared secret header check in front of the tool handlers.

None of that is needed for a demo, and all of it is work. Do it only if the
endpoint is going to outlive tomorrow.
