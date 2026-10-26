"""Runtime settings, read once from environment variables (prefix HJS_)."""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return default if value in (None, '') else value


def _env_bool(name: str, default: bool) -> bool:
    value = _env(name)
    if value is None:
        return default
    return value.strip().lower() in ('1', 'true', 'yes', 'on')


def _load_or_create_secret(data_dir: Path) -> str:
    """Dev convenience: persist a random secret so tokens survive restarts."""
    path = data_dir / '.secret_key'
    if path.exists():
        return path.read_text().strip()
    data_dir.mkdir(parents=True, exist_ok=True)
    key = secrets.token_urlsafe(48)
    path.write_text(key)
    path.chmod(0o600)
    return key


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    database_url: str
    storage_dir: Path
    secret_key: str
    # URL GPU workers use to reach this server (must be publicly reachable for RunPod).
    public_url: str
    # 'runpod' rents GPUs; 'local' runs the worker as a subprocess on this host.
    launcher: str
    runpod_api_key: str | None
    worker_image: str
    worker_python: str
    worker_mock: bool
    allow_registration: bool
    max_upload_bytes: int
    default_max_hours: float
    max_hours_limit: float
    heartbeat_timeout_s: int
    session_idle_minutes: int
    max_concurrent_jobs: int
    container_disk_gb: int
    volume_gb: int
    reaper_interval_s: int
    seed_catalog: bool
    cors_origins: list[str] = field(default_factory=list)


@lru_cache
def get_settings() -> Settings:
    data_dir = Path(_env('HJS_DATA_DIR', './data')).resolve()
    secret = _env('HJS_SECRET_KEY') or _load_or_create_secret(data_dir)
    return Settings(
        data_dir=data_dir,
        database_url=_env('HJS_DATABASE_URL', f'sqlite:///{data_dir / "studio.db"}'),
        storage_dir=Path(_env('HJS_STORAGE_DIR', str(data_dir / 'storage'))).resolve(),
        secret_key=secret,
        public_url=_env('HJS_PUBLIC_URL', 'http://127.0.0.1:8000').rstrip('/'),
        launcher=_env('HJS_LAUNCHER', 'local'),
        runpod_api_key=_env('RUNPOD_API_KEY'),
        worker_image=_env('HJS_WORKER_IMAGE', 'ghcr.io/samithaj/hjepa-worker:latest'),
        worker_python=_env('HJS_WORKER_PYTHON', 'python3'),
        worker_mock=_env_bool('HJS_WORKER_MOCK', False),
        allow_registration=_env_bool('HJS_ALLOW_REGISTRATION', True),
        max_upload_bytes=int(float(_env('HJS_MAX_UPLOAD_GB', '100')) * 1024**3),
        default_max_hours=float(_env('HJS_DEFAULT_MAX_HOURS', '6')),
        max_hours_limit=float(_env('HJS_MAX_HOURS_LIMIT', '48')),
        heartbeat_timeout_s=int(_env('HJS_HEARTBEAT_TIMEOUT_S', '900')),
        session_idle_minutes=int(_env('HJS_SESSION_IDLE_MINUTES', '20')),
        max_concurrent_jobs=int(_env('HJS_MAX_CONCURRENT_JOBS', '3')),
        container_disk_gb=int(_env('HJS_CONTAINER_DISK_GB', '60')),
        volume_gb=int(_env('HJS_VOLUME_GB', '0')),
        reaper_interval_s=int(_env('HJS_REAPER_INTERVAL_S', '30')),
        seed_catalog=_env_bool('HJS_SEED_CATALOG', True),
        cors_origins=[o for o in (_env('HJS_CORS_ORIGINS', '') or '').split(',') if o],
    )
