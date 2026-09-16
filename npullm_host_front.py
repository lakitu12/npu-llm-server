#!/usr/bin/env python3
"""npullm_host_front.py — host-side OpenAI-compatible HTTP front for the NPU queue worker.

Architecture (why three pieces):
  phone App (untrusted_app SELinux) CANNOT dlopen libneuronusdk_adapter.mtk.so
  (/vendor/lib64 not in App public namespace) and CANNOT exec files staged in
  /data/local/tmp (shell_data_file: execute denied). `adb shell` (shell context)
  CAN do both. So:
    phone: /data/local/tmp/npullm/daemon.sh  (queue worker, shell ctx, real NPU)
    host:  this file                          (HTTP on :18080, polls via adb)
    App:   kept as fallback CPU path + UI status (optional)

Queue protocol (dir /data/local/tmp/npullm/queue/):
  host: printf prompt > req.<id>.in ; mv to req.<id>   (atomic submit)
  phone: lm_shell ... > resp.<id>.tmp ; mv to resp.<id> (atomic reply)
  host: poll `cat resp.<id>`, then `rm -f resp.<id>`

Usage:
  python3 npullm_host_front.py [--port 18080] [--timeout 300]
PC clients:  base_url http://127.0.0.1:18080/v1  (no adb reverse needed;
             the server runs ON the PC). Or: adb reverse tcp:18080 tcp:18080
             to expose phone-side... not needed here.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

Q = "/data/local/tmp/npullm/queue"
ADB = ["adb"]
_req_seq = [0]


def sh(*args, timeout=60):
    p = subprocess.run(list(args), capture_output=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def adb_shell(cmd, timeout=60):
    return sh(*(ADB + ["shell", cmd]), timeout=timeout)


def submit(prompt, timeout=300):
    _req_seq[0] += 1
    rid = "%d-%d" % (int(time.time() * 1000), _req_seq[0])
    with tempfile.NamedTemporaryFile("w", suffix=".in", delete=False,
                                     encoding="utf-8") as f:
        f.write(prompt[:2000])
        tmp = f.name
    # push bytes exactly (avoid shell quoting hell): base64 path
    import base64
    with open(tmp, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    os.unlink(tmp)
    # write via printf to avoid echo interpretation; chunk to stay under limits
    rc, out, err = adb_shell(
        "base64 -d > %s/req.%s.in <<'B64EOF'\n%s\nB64EOF\n"
        "mv %s/req.%s.in %s/req.%s" % (Q, rid, b64, Q, rid, Q, rid),
        timeout=60)
    if rc != 0:
        raise RuntimeError("submit failed: %s" % err.decode()[:300])
    t0 = time.time()
    while time.time() - t0 < timeout:
        rc, out, err = adb_shell("cat %s/resp.%s 2>/dev/null" % (Q, rid),
                                 timeout=30)
        if rc == 0 and out:
            adb_shell("rm -f %s/resp.%s" % (Q, rid), timeout=30)
            return out.decode("utf-8", "replace")
        time.sleep(2)
    raise TimeoutError("npu timeout after %ss (daemon alive? check daemon.log)"
                       % timeout)


def extract_prompt(d):
    try:
        msgs = d.get("messages")
        if isinstance(msgs, list) and msgs:
            c = msgs[-1].get("content", "")
            if isinstance(c, list):
                return " ".join(
                    x.get("text", "") for x in c if isinstance(x, dict))
            if isinstance(c, str):
                return c
        return d.get("prompt", "") or d.get("input", "") or ""
    except Exception:
        return ""


class H(BaseHTTPRequestHandler):
    server_version = "npullm/1.0"

    def _send(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *a):  # noqa: A002 (stdlib signature)
        sys.stderr.write("%s %s\n" % (self.command, self.path))

    def do_GET(self):
        if self.path in ("/", "/health"):
            self._send(200, {"status": "ready (shell-npu)",
                             "backend": "shell-npu", "model": "mt6991"})
        elif self.path == "/v1/models":
            self._send(200, {"object": "list", "data": [
                {"id": "npu-llm", "object": "model",
                 "owned_by": "mt6991-npu"}]})
        else:
            self._send(404, {"error": "not found: %s" % self.path})

    def do_POST(self):
        if self.path not in ("/v1/chat/completions", "/v1/completions"):
            self._send(404, {"error": "not found: %s" % self.path})
            return
        try:
            ln = int(self.headers.get("Content-Length", 0))
        except Exception:
            ln = 0
        try:
            d = json.loads(self.rfile.read(ln) or b"{}")
        except Exception:
            d = {}
        prompt = (extract_prompt(d) or "你好")[:2000]
        stream = bool(d.get("stream"))
        try:
            reply = submit(prompt)
        except Exception as e:
            self._send(500, {"error": str(e)[:500]})
            return
        if self.path == "/v1/chat/completions":
            if stream:
                chunk = {"id": "chatcmpl-npu", "object": "chat.completion.chunk",
                         "choices": [{"delta": {"content": reply},
                                      "index": 0, "finish_reason": None}]}
                payload = ("data: %s\n\ndata: [DONE]\n"
                           % json.dumps(chunk, ensure_ascii=False)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            else:
                self._send(200, {"id": "chatcmpl-npu",
                                 "object": "chat.completion", "model": "npu-llm",
                                 "choices": [{"index": 0, "message":
                                              {"role": "assistant",
                                               "content": reply},
                                              "finish_reason": "stop"}]})
        else:
            self._send(200, {"id": "cmpl-npu", "object": "text_completion",
                             "model": "npu-llm",
                             "choices": [{"text": reply, "index": 0,
                                          "finish_reason": "stop"}]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--timeout", type=int, default=300)
    a = ap.parse_args()
    import __main__  # noqa: F401  (kept for future timeout plumbing)
    # wrap submit default timeout
    global submit
    _submit = submit

    def submit2(prompt, timeout=None):
        return _submit(prompt, timeout or a.timeout)
    globals()["submit"] = submit2
    rc, out, _ = adb_shell("ls %s >/dev/null 2>&1 && echo OK" % Q)
    if b"OK" not in out:
        sys.exit("queue dir %s missing on device (daemon running?)" % Q)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    print("npullm host front on http://127.0.0.1:%d/v1 (queue=%s)"
          % (a.port, Q), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
