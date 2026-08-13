#!/usr/bin/env python3
"""Static file server for web/ with COOP/COEP headers so onnxruntime-web can use
threaded+SIMD WASM (SharedArrayBuffer needs a cross-origin-isolated context).
Run from inside web/: `python3 serve.py [port]` (default 8000)."""
import http.server
import socketserver
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()


class ThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


with ThreadingServer(("", PORT), Handler) as httpd:
    print(f"Serving web/ at http://localhost:{PORT} (COOP/COEP enabled)")
    httpd.serve_forever()
