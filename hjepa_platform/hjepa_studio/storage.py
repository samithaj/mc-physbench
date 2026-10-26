"""Blob storage on the local filesystem (mount a persistent volume at HJS_STORAGE_DIR).

Keys are relative POSIX paths such as ``datasets/<id>/data.h5``. GPU workers never
touch this directly: they get short-lived signed URLs served by ``routers/files.py``.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from .config import get_settings

_KEY_RE = re.compile(r'^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$')


class StorageError(ValueError):
    pass


def validate_key(key: str) -> str:
    if not _KEY_RE.match(key) or any(part in ('.', '..') for part in key.split('/')):
        raise StorageError(f'invalid storage key: {key!r}')
    return key


class LocalStorage:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        path = (self.root / validate_key(key)).resolve()
        if self.root not in path.parents:
            raise StorageError(f'key escapes storage root: {key!r}')
        return path

    def exists(self, key: str) -> bool:
        return self.path(key).is_file()

    def size(self, key: str) -> int:
        path = self.path(key)
        return path.stat().st_size if path.exists() else 0

    def write_bytes(self, key: str, data: bytes) -> int:
        path = self.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return len(data)

    def open_append(self, key: str):
        path = self.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        return open(path, 'ab')

    def open_write(self, key: str):
        path = self.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        return open(path, 'wb')

    def truncate(self, key: str, size: int) -> None:
        with open(self.path(key), 'r+b') as f:
            f.truncate(size)

    def delete(self, key: str) -> None:
        path = self.path(key)
        if path.is_file():
            path.unlink()

    def delete_prefix(self, prefix: str) -> None:
        path = self.path(prefix)
        if path.is_dir():
            shutil.rmtree(path)


_storage: LocalStorage | None = None


def get_storage() -> LocalStorage:
    global _storage
    if _storage is None:
        _storage = LocalStorage(get_settings().storage_dir)
    return _storage
