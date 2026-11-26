"""Drive the real UI in headless Chrome via CDP: click Start Line, let a few
parts run, then capture a screenshot and dump the rendered text.

Used to verify the ENGLISH UI actually works end-to-end (not just static HTML).
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
OUT = Path(__file__).resolve().parent.parent / "logs" / "ui_en_running.png"
PORT = 9333


class CDP:
    def __init__(self, url: str) -> None:
        self.ws_url = url
        self._id = 0
        import json as _json
        import socket

        # Minimal RFC6455 client (no external deps).
        from urllib.parse import urlparse

        u = urlparse(url)
        self.sock = socket.create_connection((u.hostname, u.port), timeout=30)
        key = base64.b64encode(b"0123456789abcdef0123").decode()
        req = (
            f"GET {u.path} HTTP/1.1\r\nHost: {u.hostname}:{u.port}\r\n"
            f"Upgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
        )
        self.sock.sendall(req.encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            buf += self.sock.recv(4096)
        self._json = _json

    def _send_frame(self, payload: str) -> None:
        data = payload.encode()
        header = bytearray([0x81])
        n = len(data)
        if n < 126:
            header.append(0x80 | n)
        elif n < 65536:
            header.append(0x80 | 126)
            header += n.to_bytes(2, "big")
        else:
            header.append(0x80 | 127)
            header += n.to_bytes(8, "big")
        mask = b"\x11\x22\x33\x44"
        header += mask
        self.sock.sendall(bytes(header) + bytes(b ^ mask[i % 4] for i, b in enumerate(data)))

    def _recv_exact(self, n: int) -> bytes:
        out = b""
        while len(out) < n:
            chunk = self.sock.recv(n - len(out))
            if not chunk:
                raise ConnectionError("socket closed")
            out += chunk
        return out

    def recv_frame(self) -> str:
        b1, b2 = self._recv_exact(2)
        ln = b2 & 0x7F
        if ln == 126:
            ln = int.from_bytes(self._recv_exact(2), "big")
        elif ln == 127:
            ln = int.from_bytes(self._recv_exact(8), "big")
        return self._recv_exact(ln).decode("utf-8", "replace")

    def call(self, method: str, params: dict | None = None, timeout: float = 60.0) -> dict:
        self._id += 1
        mid = self._id
        self._send_frame(self._json.dumps({"id": mid, "method": method, "params": params or {}}))
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self._json.loads(self.recv_frame())
            if msg.get("id") == mid:
                return msg
        raise TimeoutError(method)


def main() -> int:
    proc = subprocess.Popen(
        [
            CHROME,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--force-device-scale-factor=1",
            "--window-size=1500,1250",
            f"--remote-debugging-port={PORT}",
            "--remote-allow-origins=*",
            "--user-data-dir=" + str(Path(__file__).resolve().parent / "_chrome_profile"),
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        ws = None
        for _ in range(40):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json", timeout=2))
                page = next(t for t in tabs if t["type"] == "page")
                ws = page["webSocketDebuggerUrl"]
                break
            except Exception:
                time.sleep(0.5)
        if not ws:
            print("FAILED: could not attach to Chrome")
            return 1

        cdp = CDP(ws)
        cdp.call("Page.enable")
        cdp.call("Runtime.enable")
        cdp.call("Page.navigate", {"url": "http://127.0.0.1:8000/"})
        time.sleep(6)

        # Fast cycle so several parts complete quickly.
        cdp.call(
            "Runtime.evaluate",
            {"expression": "document.getElementById('cycle').value=0.5;'ok'", "returnByValue": True},
        )
        cdp.call(
            "Runtime.evaluate",
            {"expression": "document.getElementById('btn-start').click();'started'", "returnByValue": True},
        )

        # Wait until at least 5 parts are inspected.
        for _ in range(60):
            time.sleep(1)
            r = cdp.call(
                "Runtime.evaluate",
                {
                    "expression": "document.getElementById('s-total').textContent",
                    "returnByValue": True,
                },
            )
            total = r["result"]["result"].get("value", "0")
            if total.isdigit() and int(total) >= 5:
                break

        # Freeze the line so the screenshot is stable.
        cdp.call("Runtime.evaluate", {"expression": "document.getElementById('btn-stop').click();'stopped'"})
        time.sleep(1)

        shot = cdp.call("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": True})
        OUT.write_bytes(base64.b64decode(shot["result"]["data"]))

        # Dump what the operator would read.
        r = cdp.call(
            "Runtime.evaluate",
            {
                "expression": (
                    "JSON.stringify({"
                    "total:document.getElementById('s-total').textContent,"
                    "ok:document.getElementById('s-ok').textContent,"
                    "ng:document.getElementById('s-ng').textContent,"
                    "rate:document.getElementById('s-rate').textContent,"
                    "avg:document.getElementById('s-avg').textContent,"
                    "verdict:document.getElementById('vtext').textContent,"
                    "vmeta:document.getElementById('vmeta').innerText,"
                    "rows:document.querySelectorAll('#hist-body tr').length"
                    "})"
                ),
                "returnByValue": True,
            },
        )
        print(json.dumps(json.loads(r["result"]["result"]["value"]), indent=2))
        print(f"screenshot -> {OUT}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
