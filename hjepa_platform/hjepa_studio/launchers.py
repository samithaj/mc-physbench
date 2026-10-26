"""Start and stop GPU workers: RunPod pods in production, local subprocesses in dev."""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .config import get_settings
from .presets import GPU_CATALOG, gpu_entry
from .runpod_client import RunPodClient, RunPodError, proxy_url

log = logging.getLogger(__name__)

WORKER_DIR = Path(os.environ.get('HJS_WORKER_DIR') or Path(__file__).resolve().parents[1] / 'worker')
SESSION_PORT = 8000


class LaunchError(RuntimeError):
    pass


@dataclass
class LaunchResult:
    backend: str
    pod_id: str
    cost_per_hr: float = 0.0
    endpoint_url: str | None = None
    info: dict = field(default_factory=dict)


def _worker_env(mode: str, ident: str, token: str) -> dict[str, str]:
    settings = get_settings()
    env = {
        'HJW_PLATFORM_URL': settings.public_url,
        'HJW_MODE': mode,
        'HJW_ID': ident,
        'HJW_TOKEN': token,
    }
    if settings.worker_mock:
        env['HJW_MOCK'] = '1'
    return env


class RunPodLauncher:
    name = 'runpod'

    def launch(self, *, mode: str, ident: str, token: str, gpu: dict, api_key: str | None) -> LaunchResult:
        settings = get_settings()
        client = RunPodClient(api_key or '')
        # Fall back to similar small GPUs if the first choice is out of stock.
        first = gpu.get('type')
        fallbacks = [g['id'] for g in GPU_CATALOG if g['id'] != first and gpu.get('allow_fallback')]
        env = _worker_env(mode, ident, token)
        ports = [f'{SESSION_PORT}/http'] if mode == 'serve' else []
        if mode == 'serve':
            env['HJW_PORT'] = str(SESSION_PORT)
        try:
            pod = client.create_pod(
                name=f'hjepa-{mode}-{ident}',
                image=settings.worker_image,
                gpu_type_ids=[first, *fallbacks],
                env=env,
                gpu_count=int(gpu.get('count', 1)),
                cloud_type=gpu.get('cloud_type', 'SECURE'),
                container_disk_gb=settings.container_disk_gb,
                volume_gb=settings.volume_gb,
                ports=ports,
            )
        except RunPodError as exc:
            raise LaunchError(str(exc)) from exc
        cost = pod.get('costPerHr')
        if cost is None:  # estimate from the catalog until the reaper sees the real price
            cost = (gpu_entry(first) or {}).get('est_usd_hr', 0.0) * int(gpu.get('count', 1))
        return LaunchResult(
            backend=self.name,
            pod_id=pod['id'],
            cost_per_hr=float(cost),
            endpoint_url=proxy_url(pod['id'], SESSION_PORT) if mode == 'serve' else None,
            info={'machine': pod.get('machine'), 'gpu': (pod.get('gpu') or {}).get('id', first)},
        )

    def terminate(self, pod_id: str, api_key: str | None) -> None:
        try:
            RunPodClient(api_key or '').terminate_pod(pod_id)
        except RunPodError as exc:
            log.warning('terminate %s failed: %s', pod_id, exc)

    def is_alive(self, pod_id: str, api_key: str | None) -> bool | None:
        """True/False when known, None when the API could not be reached."""
        try:
            pod = RunPodClient(api_key or '').get_pod(pod_id)
        except RunPodError:
            return None
        if pod is None:
            return False
        return pod.get('desiredStatus') not in ('EXITED', 'TERMINATED')


class LocalLauncher:
    """Runs the worker agent as a subprocess of the server (dev, CI, or a GPU box)."""

    name = 'local'

    def __init__(self):
        self._procs: dict[str, subprocess.Popen] = {}

    @staticmethod
    def _free_port() -> int:
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0))
            return s.getsockname()[1]

    def launch(self, *, mode: str, ident: str, token: str, gpu: dict, api_key: str | None) -> LaunchResult:
        settings = get_settings()
        env = {**os.environ, **_worker_env(mode, ident, token)}
        env['PYTHONPATH'] = str(WORKER_DIR) + os.pathsep + env.get('PYTHONPATH', '')
        endpoint = None
        if mode == 'serve':
            port = self._free_port()
            env['HJW_PORT'] = str(port)
            env['HJW_HOST'] = '127.0.0.1'
            endpoint = f'http://127.0.0.1:{port}'
        log_dir = settings.data_dir / 'worker_logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = open(log_dir / f'{mode}-{ident}.log', 'ab')
        python = settings.worker_python if settings.worker_python != 'python3' else sys.executable
        try:
            proc = subprocess.Popen(
                [python, '-m', 'hjepa_worker.agent'],
                cwd=str(WORKER_DIR), env=env, stdout=log_file, stderr=subprocess.STDOUT,
            )
        except OSError as exc:
            raise LaunchError(f'could not start local worker: {exc}') from exc
        finally:
            log_file.close()
        pod_id = f'local-{proc.pid}'
        self._procs[pod_id] = proc
        return LaunchResult(backend=self.name, pod_id=pod_id, endpoint_url=endpoint,
                            info={'gpu': gpu.get('type'), 'pid': proc.pid})

    def terminate(self, pod_id: str, api_key: str | None) -> None:
        proc = self._procs.pop(pod_id, None)
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    def is_alive(self, pod_id: str, api_key: str | None) -> bool | None:
        proc = self._procs.get(pod_id)
        if proc is None:
            return None  # started by a previous server process; unknown
        return proc.poll() is None


_launchers: dict[str, object] = {}


def get_launcher(name: str | None = None):
    name = name or get_settings().launcher
    if name not in _launchers:
        if name == 'runpod':
            _launchers[name] = RunPodLauncher()
        elif name == 'local':
            _launchers[name] = LocalLauncher()
        else:
            raise LaunchError(f'unknown launcher {name!r}')
    return _launchers[name]
