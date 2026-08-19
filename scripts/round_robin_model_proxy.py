#!/usr/bin/env python3
"""Round-robin proxy for multiple OpenAI-compatible local model servers."""

from __future__ import annotations

import argparse
import json
import threading
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", action="append", required=True, help="Backend base URL, repeatable.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7999)
    parser.add_argument("--timeout-sec", type=float, default=300.0)
    return parser.parse_args()


class BackendPool:
    def __init__(self, backends: list[str]) -> None:
        if not backends:
            raise ValueError("at least one backend is required")
        self.backends = [normalize_backend(value) for value in backends]
        self._index = 0
        self._lock = threading.Lock()

    def next_order(self) -> list[str]:
        with self._lock:
            start = self._index
            self._index = (self._index + 1) % len(self.backends)
        return self.backends[start:] + self.backends[:start]


def normalize_backend(value: str) -> str:
    return value.rstrip("/")


def fetch_json(url: str, timeout_sec: float) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return response.status, payload
    except urllib.error.HTTPError as exc:
        payload = json.loads(exc.read().decode("utf-8"))
        return exc.code, payload


def build_handler(pool: BackendPool, timeout_sec: float) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "LottieRoundRobin/1.0"

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self._write_json(HTTPStatus.NOT_FOUND, {"error": {"message": "route not found"}})
                return
            checks = []
            for backend in pool.backends:
                try:
                    status, payload = fetch_json(f"{backend}/health", min(timeout_sec, 10.0))
                    checks.append({"backend": backend, "status": status, "payload": payload})
                except Exception as exc:
                    checks.append({"backend": backend, "status": None, "error": repr(exc)})
            healthy = [item for item in checks if item.get("status") == HTTPStatus.OK]
            status = HTTPStatus.OK if len(healthy) == len(checks) else HTTPStatus.SERVICE_UNAVAILABLE
            self._write_json(status, {"status": "ok" if status == HTTPStatus.OK else "degraded", "backends": checks})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/chat/completions":
                self._write_json(HTTPStatus.NOT_FOUND, {"error": {"message": "route not found"}})
                return
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            errors = []
            for backend in pool.next_order():
                request = urllib.request.Request(
                    f"{backend}{self.path}",
                    data=body,
                    headers={
                        "Content-Type": self.headers.get("Content-Type") or "application/json",
                        "Authorization": self.headers.get("Authorization") or "Bearer local",
                    },
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
                        self._write_bytes(response.status, response.read(), response.headers.get("Content-Type"))
                        return
                except urllib.error.HTTPError as exc:
                    self._write_bytes(exc.code, exc.read(), exc.headers.get("Content-Type"))
                    return
                except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                    errors.append({"backend": backend, "error": repr(exc)})
            self._write_json(
                HTTPStatus.BAD_GATEWAY,
                {"error": {"message": "all model backends unavailable", "backends": errors}},
            )

        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def _write_json(self, status: HTTPStatus | int, payload: dict[str, Any]) -> None:
            self._write_bytes(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json")

        def _write_bytes(self, status: HTTPStatus | int, body: bytes, content_type: str | None) -> None:
            self.send_response(int(status))
            self.send_header("Content-Type", content_type or "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main() -> None:
    args = parse_args()
    pool = BackendPool(args.backend)
    server = ThreadingHTTPServer((args.host, args.port), build_handler(pool, args.timeout_sec))
    print(
        json.dumps(
            {"status": "serving", "host": args.host, "port": args.port, "backends": pool.backends},
            ensure_ascii=False,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
