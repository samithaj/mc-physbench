"""Signed-URL file transfer: browsers and workers download, workers upload artifacts."""

from __future__ import annotations

import mimetypes

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import JobArtifact
from ..security import verify_file_signature
from ..storage import StorageError, get_storage

router = APIRouter(prefix='/api/files', tags=['files'])

MAX_ARTIFACT_BYTES = 20 * 1024**3


def _check(method: str, key: str, exp: int, sig: str) -> None:
    if not verify_file_signature(method, key, exp, sig):
        raise HTTPException(status.HTTP_403_FORBIDDEN, 'invalid or expired link')


@router.get('/{key:path}')
def download(key: str, exp: int = Query(), sig: str = Query()):
    _check('GET', key, exp, sig)
    storage = get_storage()
    try:
        path = storage.path(key)
    except StorageError:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, 'bad key')
    if not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, 'not found')
    media = mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
    return FileResponse(path, media_type=media, filename=path.name if media.startswith('application') else None)


@router.put('/{key:path}')
async def upload(key: str, request: Request, exp: int = Query(), sig: str = Query(),
                 db: Session = Depends(get_db)):
    _check('PUT', key, exp, sig)
    if not key.startswith('jobs/'):
        raise HTTPException(status.HTTP_403_FORBIDDEN, 'uploads are only allowed for job artifacts')
    storage = get_storage()
    try:
        tmp_key = key + '.part'
        written = 0
        with storage.open_write(tmp_key) as fh:
            async for chunk in request.stream():
                written += len(chunk)
                if written > MAX_ARTIFACT_BYTES:
                    raise HTTPException(413, 'artifact too large')
                fh.write(chunk)
        storage.path(tmp_key).replace(storage.path(key))
    except StorageError:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, 'bad key')
    row = db.scalar(select(JobArtifact).where(JobArtifact.storage_key == key))
    if row is not None:
        row.size_bytes = written
        db.commit()
    return {'key': key, 'size_bytes': written}
