from __future__ import annotations

import re

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db import get_db
from ..deps import current_user, get_owned, get_readable
from ..h5inspect import inspect_h5
from ..models import Dataset, User
from ..presets import ENV_PRESETS
from ..security import signed_file_url
from ..serializers import dataset_dict
from ..storage import get_storage

router = APIRouter(prefix='/api/datasets', tags=['datasets'])

MAX_CHUNK = 64 * 1024 * 1024
_HF_REPO_RE = re.compile(r'^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$')
_HF_FILE_RE = re.compile(r'^[A-Za-z0-9_./-]+\.h5$')


class DatasetMeta(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    description: str = Field(default='', max_length=5000)
    visibility: str = Field(default='private', pattern='^(private|public)$')
    env_preset: str = 'custom'
    license: str = Field(default='', max_length=64)
    tags: list[str] = Field(default_factory=list, max_length=12)


class CreateUpload(DatasetMeta):
    size_bytes: int = Field(gt=0)


class ImportHF(DatasetMeta):
    repo: str
    filename: str
    repo_type: str = Field(default='dataset', pattern='^(dataset|model)$')
    revision: str | None = Field(default=None, max_length=64)


class PatchDataset(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=128)
    description: str | None = Field(default=None, max_length=5000)
    visibility: str | None = Field(default=None, pattern='^(private|public)$')
    env_preset: str | None = None
    license: str | None = Field(default=None, max_length=64)
    tags: list[str] | None = Field(default=None, max_length=12)


def _check_preset(preset: str | None) -> None:
    if preset is not None and preset not in ENV_PRESETS:
        raise HTTPException(422, f'unknown env preset {preset!r}')


def _clean_tags(tags: list[str]) -> list[str]:
    return [t.strip()[:32] for t in tags if t.strip()]


@router.get('')
def list_datasets(scope: str = Query('all', pattern='^(mine|public|all)$'),
                  preset: str | None = None, q: str | None = None,
                  user: User = Depends(current_user), db: Session = Depends(get_db)):
    stmt = select(Dataset)
    if scope == 'mine':
        stmt = stmt.where(Dataset.owner_id == user.id)
    elif scope == 'public':
        stmt = stmt.where(Dataset.visibility == 'public')
    else:
        stmt = stmt.where(or_(Dataset.owner_id == user.id, Dataset.visibility == 'public'))
    if preset:
        stmt = stmt.where(Dataset.env_preset == preset)
    if q:
        stmt = stmt.where(Dataset.name.ilike(f'%{q}%'))
    rows = db.scalars(stmt.order_by(Dataset.created_at.desc()).limit(200)).all()
    return [dataset_dict(d, user) for d in rows]


@router.post('', status_code=201)
def create_upload(body: CreateUpload, user: User = Depends(current_user),
                  db: Session = Depends(get_db)):
    _check_preset(body.env_preset)
    if body.size_bytes > get_settings().max_upload_bytes:
        raise HTTPException(413, 'file is too large')
    ds = Dataset(owner_id=user.id, name=body.name, description=body.description,
                 visibility=body.visibility, env_preset=body.env_preset, license=body.license,
                 tags=_clean_tags(body.tags), source='upload', status='uploading',
                 expected_bytes=body.size_bytes)
    db.add(ds)
    db.flush()
    ds.storage_key = f'datasets/{ds.id}/data.h5'
    get_storage().write_bytes(ds.storage_key, b'')
    db.commit()
    return dataset_dict(ds, user)


@router.post('/import-hf', status_code=201)
def import_hf(body: ImportHF, user: User = Depends(current_user), db: Session = Depends(get_db)):
    _check_preset(body.env_preset)
    if not _HF_REPO_RE.match(body.repo) or not _HF_FILE_RE.match(body.filename) \
            or '..' in body.filename:
        raise HTTPException(422,
                            'repo must be owner/name and filename a relative .h5 path')
    ds = Dataset(owner_id=user.id, name=body.name, description=body.description,
                 visibility=body.visibility, env_preset=body.env_preset, license=body.license,
                 tags=_clean_tags(body.tags), source='huggingface', status='ready',
                 source_ref={'repo': body.repo, 'filename': body.filename,
                             'repo_type': body.repo_type, 'revision': body.revision},
                 inspection={'valid': None,
                             'note': 'Hugging Face files are downloaded and validated on the GPU worker.'})
    db.add(ds)
    db.commit()
    return dataset_dict(ds, user)


@router.get('/{dataset_id}')
def get_dataset(dataset_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    return dataset_dict(get_readable(db, Dataset, dataset_id, user), user)


@router.get('/{dataset_id}/upload')
def upload_status(dataset_id: str, user: User = Depends(current_user),
                  db: Session = Depends(get_db)):
    ds = get_owned(db, Dataset, dataset_id, user)
    return {'received': get_storage().size(ds.storage_key) if ds.storage_key else 0,
            'expected': ds.expected_bytes, 'status': ds.status}


@router.put('/{dataset_id}/upload')
async def upload_chunk(dataset_id: str, request: Request, offset: int = Query(ge=0),
                       user: User = Depends(current_user), db: Session = Depends(get_db)):
    ds = get_owned(db, Dataset, dataset_id, user)
    if ds.source != 'upload' or ds.status != 'uploading':
        raise HTTPException(status.HTTP_409_CONFLICT, 'dataset is not accepting uploads')
    storage = get_storage()
    received = storage.size(ds.storage_key)
    if offset != received:
        raise HTTPException(status.HTTP_409_CONFLICT,
                            detail={'message': 'offset mismatch', 'received': received})
    written = 0
    with storage.open_append(ds.storage_key) as fh:
        async for chunk in request.stream():
            written += len(chunk)
            if written > MAX_CHUNK or received + written > ds.expected_bytes:
                fh.close()
                storage.truncate(ds.storage_key, received)
                raise HTTPException(413,
                                    'chunk too large or file larger than declared')
            fh.write(chunk)
    return {'received': received + written, 'expected': ds.expected_bytes}


@router.post('/{dataset_id}/complete')
async def complete_upload(dataset_id: str, user: User = Depends(current_user),
                          db: Session = Depends(get_db)):
    ds = get_owned(db, Dataset, dataset_id, user)
    if ds.source != 'upload' or ds.status not in ('uploading', 'invalid'):
        raise HTTPException(status.HTTP_409_CONFLICT, 'dataset is not awaiting completion')
    storage = get_storage()
    size = storage.size(ds.storage_key)
    if size != ds.expected_bytes:
        raise HTTPException(status.HTTP_409_CONFLICT,
                            f'upload incomplete: {size} of {ds.expected_bytes} bytes')
    ds.status = 'processing'
    db.commit()
    report, preview = await run_in_threadpool(inspect_h5, storage.path(ds.storage_key))
    ds.size_bytes = size
    ds.inspection = report
    ds.status = 'ready' if report['valid'] else 'invalid'
    if preview:
        ds.preview_key = f'datasets/{ds.id}/preview.png'
        storage.write_bytes(ds.preview_key, preview)
    compatible = report.get('compatible_presets') or []
    if report['valid'] and ds.env_preset not in compatible:
        # Built-in configs need specific columns; anything else trains as custom data.
        ds.env_preset = 'custom'
    db.commit()
    return dataset_dict(ds, user)


@router.patch('/{dataset_id}')
def patch_dataset(dataset_id: str, body: PatchDataset, user: User = Depends(current_user),
                  db: Session = Depends(get_db)):
    ds = get_owned(db, Dataset, dataset_id, user)
    _check_preset(body.env_preset)
    if body.env_preset and ds.source == 'upload' and ds.inspection.get('valid'):
        if body.env_preset not in (ds.inspection.get('compatible_presets') or []):
            raise HTTPException(422,
                                f'data is not compatible with preset {body.env_preset!r}')
    for field in ('name', 'description', 'visibility', 'env_preset', 'license'):
        value = getattr(body, field)
        if value is not None:
            setattr(ds, field, value)
    if body.tags is not None:
        ds.tags = _clean_tags(body.tags)
    db.commit()
    return dataset_dict(ds, user)


@router.delete('/{dataset_id}', status_code=204)
def delete_dataset(dataset_id: str, user: User = Depends(current_user),
                   db: Session = Depends(get_db)):
    ds = get_owned(db, Dataset, dataset_id, user)
    if ds.source == 'upload':
        get_storage().delete_prefix(f'datasets/{ds.id}')
    db.delete(ds)
    db.commit()


@router.get('/{dataset_id}/download')
def download_dataset(dataset_id: str, user: User = Depends(current_user),
                     db: Session = Depends(get_db)):
    ds = get_readable(db, Dataset, dataset_id, user)
    if ds.source != 'upload' or ds.status != 'ready':
        raise HTTPException(status.HTTP_400_BAD_REQUEST, 'only uploaded, ready datasets can be downloaded')
    ds.use_count += 1
    db.commit()
    return {'url': signed_file_url(ds.storage_key, ttl_s=3600)}
