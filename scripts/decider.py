"""The decider from the command line, and as a small HTTP service (stdlib only).

    uv run python scripts/decider.py bootstrap --source sst        # train from the B3 cache, save
    uv run python scripts/decider.py decide --text "I fucking hate you"
    uv run python scripts/decider.py decide --image data/raw/oasis/images/"Dessert 1.jpg"
    uv run python scripts/decider.py reward --id <id> --taste bitter
    uv run python scripts/decider.py state
    uv run python scripts/decider.py serve --port 8765
        POST /decide  {"text": ..., "image": ...}     POST /reward {"id": ..., "taste": "sweet"|"bitter"|-1..1}
        GET  /state

State lives in state/decider/ (or --state DIR) and is saved after every change.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bosco import paths
from bosco.decider import Decider


def open_decider(a) -> Decider:
    d = Path(a.state)
    return Decider.load(d) if (d / "meta.json").exists() else Decider(a.arm, state_dir=d)


def cmd_bootstrap(a) -> int:
    d = open_decider(a)
    print(json.dumps(d.bootstrap(a.source, a.n_per_side, a.seed), indent=1))
    d.save()
    return 0


def cmd_decide(a) -> int:
    d = open_decider(a)
    r = d.decide(text=a.text, image=a.image)
    print(json.dumps(r, indent=1))
    d.save()
    return 0


def cmd_reward(a) -> int:
    d = open_decider(a)
    taste = a.taste
    try:
        taste = float(taste)
    except ValueError:
        pass
    print(json.dumps(d.reward(a.id, taste, a.magnitude), indent=1))
    d.save()
    return 0


def cmd_state(a) -> int:
    print(json.dumps(open_decider(a).state(), indent=1))
    return 0


def cmd_serve(a) -> int:
    from http.server import BaseHTTPRequestHandler, HTTPServer

    d = open_decider(a)

    class H(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            if self.path == "/state":
                return self._send(200, d.state())
            return self._send(404, {"error": "GET /state"})

        def do_POST(self):  # noqa: N802
            n = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
                if self.path == "/decide":
                    r = d.decide(text=body.get("text"), image=body.get("image"))
                elif self.path == "/reward":
                    r = d.reward(body["id"], body.get("taste", "sweet"), float(body.get("magnitude", 1.0)))
                else:
                    return self._send(404, {"error": "POST /decide or /reward"})
                d.save()
                return self._send(200, r)
            except (KeyError, ValueError, FileNotFoundError) as e:
                return self._send(400, {"error": str(e)})

        def log_message(self, fmt, *args):  # quiet
            sys.stderr.write(f"{self.address_string()} - {fmt % args}\n")

    print(f"decider ({d.arm}) on http://127.0.0.1:{a.port}  POST /decide /reward, GET /state", flush=True)
    HTTPServer(("127.0.0.1", a.port), H).serve_forever()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=str(paths.STATE / "decider"))
    ap.add_argument("--arm", default="real", choices=["real", "shuffle", "hash"])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("bootstrap")
    p.add_argument("--source", default="sst", choices=["sst", "oasis"])
    p.add_argument("--n-per-side", type=int, default=200)
    p.add_argument("--seed", type=int, default=1)
    p.set_defaults(fn=cmd_bootstrap)
    p = sub.add_parser("decide")
    p.add_argument("--text")
    p.add_argument("--image")
    p.set_defaults(fn=cmd_decide)
    p = sub.add_parser("reward")
    p.add_argument("--id", required=True)
    p.add_argument("--taste", required=True)
    p.add_argument("--magnitude", type=float, default=1.0)
    p.set_defaults(fn=cmd_reward)
    sub.add_parser("state").set_defaults(fn=cmd_state)
    p = sub.add_parser("serve")
    p.add_argument("--port", type=int, default=8765)
    p.set_defaults(fn=cmd_serve)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
