"""HTTP server for an interactive simulation session (stdlib only).

Only the platform talks to it: every request must carry X-Session-Secret, which the
platform hands to the worker through the authenticated session spec.
"""

from __future__ import annotations

import hmac
import json
import os
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .client import PlatformClient, PlatformError

CALLS = ('info', 'reset', 'step', 'rollout', 'sample_action')


def build_engine(spec: dict, log=print):
    if os.environ.get('HJW_MOCK') == '1':
        from .mock import MockEngine
        return MockEngine(spec)
    from .engine import InteractiveEngine
    from .inputs import fetch_dataset, fetch_model
    from .simulator import WorldModelSimulator

    model = fetch_model(spec['inputs']['model'], log=log)
    if not spec['inputs']['datasets']:
        raise ValueError('session has no dataset to take start frames from')
    ds = fetch_dataset(spec['inputs']['datasets'][0], log=log)
    h5 = ds.get('val_path') or ds['path']
    sim = WorldModelSimulator(model['ckpt'], model['normalizer'], model['config'], model['decoders'])
    return InteractiveEngine(sim, h5, {'model_name': spec['params'].get('model_name')})


def make_handler(engine, secret: str, lock: threading.Lock):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict) -> None:
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # health probe
            self._send(200, {'ok': True})

        def do_POST(self):
            given = self.headers.get('X-Session-Secret', '')
            if not hmac.compare_digest(given, secret):
                return self._send(403, {'error': 'forbidden'})
            call = self.path.strip('/').split('?')[0]
            if call not in CALLS:
                return self._send(404, {'error': 'unknown call'})
            length = int(self.headers.get('Content-Length') or 0)
            try:
                body = json.loads(self.rfile.read(length) or b'{}') if length else {}
                with lock:  # one GPU, one model state: serialize calls
                    result = getattr(engine, call)(body)
                self._send(200, result)
            except ValueError as exc:
                self._send(400, {'error': str(exc)})
            except Exception as exc:
                traceback.print_exc()
                self._send(500, {'error': f'{type(exc).__name__}: {exc}'})

        def log_message(self, fmt, *args):
            pass

    return Handler


def serve(client: PlatformClient) -> None:
    spec = client._call('GET', '/spec')
    try:
        engine = build_engine(spec)
    except Exception as exc:
        traceback.print_exc()
        client.session_ready(failed=True, message=f'{type(exc).__name__}: {exc}')
        sys.exit(1)

    host = os.environ.get('HJW_HOST', '0.0.0.0')
    port = int(os.environ.get('HJW_PORT', '8000'))
    httpd = ThreadingHTTPServer((host, port), make_handler(engine, spec['worker_secret'],
                                                           threading.Lock()))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(f'session server on {host}:{port}', flush=True)
    reply = client.session_ready(info=engine.info({}))
    stop = threading.Event()
    if reply.get('stop'):
        stop.set()
    while not stop.wait(30):
        try:
            if client.session_heartbeat().get('stop'):
                stop.set()
        except PlatformError as exc:
            print(f'heartbeat failed: {exc}', file=sys.stderr)
    httpd.shutdown()
