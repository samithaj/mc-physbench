"""Stdlib-only HTTP client for the platform's worker API (so mock mode needs no deps)."""

from __future__ import annotations

import json
import os
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path


class PlatformError(RuntimeError):
    pass


def _request(method: str, url: str, *, token: str | None = None, body=None,
             data: bytes | None = None, headers: dict | None = None, timeout: float = 60,
             retries: int = 4):
    hdrs = dict(headers or {})
    if token:
        hdrs['Authorization'] = f'Bearer {token}'
    if body is not None:
        data = json.dumps(body).encode()
        hdrs['Content-Type'] = 'application/json'
    delay = 2.0
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, method=method, headers=hdrs)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as exc:
            # 4xx are final (bad token, cancelled job); retry server errors.
            if exc.code < 500 or attempt == retries:
                raise PlatformError(f'{method} {url} -> {exc.code}: {exc.read()[:300]!r}') from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt == retries:
                raise PlatformError(f'{method} {url} failed: {exc}') from exc
        time.sleep(delay)
        delay *= 2


def download(url: str, dest: Path, retries: int = 4) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + '.part')
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, 'wb') as fh:
                shutil.copyfileobj(resp, fh, length=8 * 1024 * 1024)
            tmp.replace(dest)
            return dest
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt == retries:
                raise PlatformError(f'download failed: {exc}') from exc
            time.sleep(2 ** (attempt + 1))
    return dest


class _FileBody:
    """Iterable file body with a known length, so urllib streams instead of buffering."""

    def __init__(self, path: Path):
        self.path = path
        self.size = path.stat().st_size

    def __iter__(self):
        with open(self.path, 'rb') as fh:
            while chunk := fh.read(8 * 1024 * 1024):
                yield chunk


class PlatformClient:
    def __init__(self, base_url: str, token: str, ident: str, mode: str = 'job'):
        self.base = base_url.rstrip('/')
        self.token = token
        self.ident = ident
        self.prefix = f'{self.base}/api/worker/{"jobs" if mode == "job" else "sessions"}/{ident}'

    @classmethod
    def from_env(cls) -> 'PlatformClient':
        mode = os.environ.get('HJW_MODE', 'job')
        return cls(os.environ['HJW_PLATFORM_URL'], os.environ['HJW_TOKEN'],
                   os.environ['HJW_ID'], 'job' if mode == 'job' else 'serve')

    def _call(self, method: str, path: str, body=None, retries: int = 4):
        return _request(method, self.prefix + path, token=self.token, body=body, retries=retries)

    # jobs
    def spec(self) -> dict:
        return self._call('GET', '/spec')

    def heartbeat(self, progress: float | None = None, message: str | None = None,
                  status: str | None = None) -> dict:
        body = {k: v for k, v in (('progress', progress), ('message', message),
                                  ('status', status)) if v is not None}
        return self._call('POST', '/heartbeat', body, retries=2) or {}

    def logs(self, lines: list[str]) -> None:
        for i in range(0, len(lines), 1000):
            self._call('POST', '/logs', {'lines': lines[i:i + 1000]}, retries=2)

    def metrics(self, rows: list[dict]) -> None:
        for i in range(0, len(rows), 2000):
            self._call('POST', '/metrics', {'rows': rows[i:i + 2000]}, retries=2)

    def upload_artifact(self, name: str, path: Path, content_type: str = 'application/octet-stream') -> str:
        info = self._call('POST', '/artifacts', {'name': name, 'content_type': content_type})
        body = _FileBody(path)
        req_headers = {'Content-Length': str(body.size), 'Content-Type': content_type}
        delay = 2.0
        for attempt in range(4):
            req = urllib.request.Request(info['upload_url'], data=body, method='PUT',
                                         headers=req_headers)
            try:
                with urllib.request.urlopen(req, timeout=600) as resp:
                    resp.read()
                return name
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                if attempt == 3:
                    raise PlatformError(f'upload of {name} failed: {exc}') from exc
                time.sleep(delay)
                delay *= 2
        return name

    def complete(self, status: str, message: str = '', result: dict | None = None) -> dict:
        return self._call('POST', '/complete', {'status': status, 'message': message[:5000],
                                                'result': result or {}})

    # sessions
    def session_ready(self, info: dict | None = None, failed: bool = False, message: str = '') -> dict:
        return self._call('POST', '/ready', {'info': info or {}, 'failed': failed,
                                             'message': message[:5000]})

    def session_heartbeat(self) -> dict:
        return self._call('POST', '/heartbeat', {}, retries=2) or {}
