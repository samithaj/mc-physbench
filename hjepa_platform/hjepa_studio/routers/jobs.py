from __future__ import annotations

import re

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db import get_db
from ..deps import current_user, get_owned, get_readable
from ..models import Dataset, Job, JobArtifact, JobLog, JobMetric, User, WorldModel
from ..presets import (DEFAULT_GPU, ENV_PRESETS, EST_TRAIN_HOURS_PER_EPOCH, GPU_IDS, VARIANTS,
                       default_planner, gpu_entry, planners_for)
from ..security import signed_file_url
from ..serializers import iso, job_dict
from ..services import (ACTIVE_JOB_STATES, check_quota, finalize_job, launch_job)

router = APIRouter(prefix='/api/jobs', tags=['jobs'])

_OVERRIDE_RE = re.compile(r'^\+?[A-Za-z_][A-Za-z0-9_.]*=[A-Za-z0-9_.:/\-\[\], ]*$')
_FORBIDDEN_OVERRIDE_PREFIXES = ('hydra', '+hydra', 'subdir', 'output_model_name', 'cache_dir')


class GpuIn(BaseModel):
    type: str = DEFAULT_GPU
    count: int = Field(default=1, ge=1, le=2)
    cloud_type: str = Field(default='SECURE', pattern='^(SECURE|COMMUNITY)$')
    allow_fallback: bool = False


class TrainIn(BaseModel):
    name: str = Field(default='', max_length=128)
    model_name: str = Field(default='', max_length=128)
    preset: str
    variant: str = 'hjepa_l2'
    train_dataset_id: str
    val_dataset_id: str | None = None
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    epochs: int = Field(default=5, ge=1, le=200)
    batch_size: int = Field(default=128, ge=4, le=1024)
    frameskip: int | None = Field(default=None, ge=1, le=50)
    run_planning_eval: bool = True
    train_decoders: bool = True
    extra_overrides: list[str] = Field(default_factory=list, max_length=24)
    output_visibility: str = Field(default='private', pattern='^(private|public)$')
    gpu: GpuIn = Field(default_factory=GpuIn)
    max_hours: float | None = Field(default=None, gt=0)


class EvaluateIn(BaseModel):
    model_id: str
    planner: str | None = None
    num_eval: int = Field(default=50, ge=1, le=50)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    gpu: GpuIn = Field(default_factory=GpuIn)
    max_hours: float | None = Field(default=None, gt=0)


class SimulateIn(BaseModel):
    model_id: str
    dataset_id: str | None = None
    episodes: list[int] = Field(default_factory=lambda: [0], min_length=1, max_length=8)
    start: int = Field(default=0, ge=0)
    context: int = Field(default=3, ge=1, le=16)
    horizon: int = Field(default=24, ge=1, le=256)
    levels: list[int] | None = None
    action_mode: str = Field(default='ground_truth', pattern='^(ground_truth|zero|random|reversed)$')
    action_scale: float = Field(default=1.0, ge=-10, le=10)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)
    gpu: GpuIn = Field(default_factory=GpuIn)
    max_hours: float | None = Field(default=None, gt=0)


def _max_hours(value: float | None) -> float:
    settings = get_settings()
    return min(value or settings.default_max_hours, settings.max_hours_limit)


def _check_gpu(gpu: GpuIn) -> dict:
    if gpu.type not in GPU_IDS:
        raise HTTPException(422, f'unsupported GPU {gpu.type!r}')
    return gpu.model_dump()


def _ready_dataset(db: Session, ds_id: str, user: User) -> Dataset:
    ds = get_readable(db, Dataset, ds_id, user)
    if ds.status != 'ready':
        raise HTTPException(422, f'dataset {ds.name!r} is not ready')
    return ds


def _dataset_fits(ds: Dataset, preset: str) -> bool:
    if ds.source == 'official':
        return ds.env_preset == preset
    if preset == 'custom':
        return True
    return ds.env_preset == preset


def _validate_train(db: Session, body: TrainIn, user: User) -> dict:
    if body.preset not in ENV_PRESETS:
        raise HTTPException(422, f'unknown preset {body.preset!r}')
    if body.variant not in VARIANTS:
        raise HTTPException(422, f'unknown variant {body.variant!r}')
    train = _ready_dataset(db, body.train_dataset_id, user)
    if not _dataset_fits(train, body.preset):
        raise HTTPException(422,
                            f'dataset {train.name!r} is tagged {train.env_preset!r}, not {body.preset!r}')
    val = None
    if body.val_dataset_id and train.source != 'official':
        val = _ready_dataset(db, body.val_dataset_id, user)
        if val.source == 'official' or not _dataset_fits(val, body.preset):
            raise HTTPException(422, 'validation dataset does not match')
        for key in ('img_size', 'action_dim'):
            a, b = train.inspection.get(key), val.inspection.get(key)
            if a is not None and b is not None and a != b:
                raise HTTPException(422,
                                    f'train and validation {key} differ ({a} vs {b})')
    for ov in body.extra_overrides:
        if not _OVERRIDE_RE.match(ov) or ov.startswith(_FORBIDDEN_OVERRIDE_PREFIXES):
            raise HTTPException(422, f'override not allowed: {ov!r}')
    preset = ENV_PRESETS[body.preset]
    official_data = train.source == 'official'
    spec = body.model_dump(exclude={'gpu', 'max_hours', 'val_dataset_id'})
    spec.update(
        val_dataset_id=val.id if val else None,
        levels=VARIANTS[body.variant]['levels'],
        frameskip=body.frameskip or preset['frameskip'],
        # Upstream planning eval needs the simulator plus the official eval tasks.
        run_planning_eval=bool(body.run_planning_eval and preset['simulator']),
        planner=default_planner(VARIANTS[body.variant]['levels']),
        official_data=official_data,
        data_info={k: train.inspection.get(k) for k in ('img_size', 'action_dim', 'has_proprio')},
        gpu=_check_gpu(body.gpu),
        max_hours=_max_hours(body.max_hours),
    )
    train.use_count += 1
    return spec


def estimate(spec: dict) -> dict:
    gpu = gpu_entry(spec['gpu']['type']) or {}
    per_hr = gpu.get('est_usd_hr', 0.0) * spec['gpu'].get('count', 1)
    if spec.get('kind', 'train') == 'train':
        hours = EST_TRAIN_HOURS_PER_EPOCH.get(spec['preset'], 0.25) * spec['epochs']
        hours *= (1 + 0.35 * (spec['levels'] - 1))  # upper levels add predictor work
        hours *= {32: 1.0, 24: 1.4, 20: 1.8, 16: 2.2}.get(gpu.get('vram_gb', 24), 1.4)
        hours /= spec['gpu'].get('count', 1)
        hours += 0.3 if spec.get('run_planning_eval') else 0.0
        hours += 0.25  # image pull + data download + decoder training
    else:
        hours = 0.5
    hours = min(hours, spec['max_hours'])
    return {'hours': round(hours, 2), 'usd_per_hr': per_hr, 'usd': round(hours * per_hr, 2),
            'note': 'rough estimate; RunPod bills per second at the live price'}


def _create(db: Session, user: User, kind: str, name: str, spec: dict) -> dict:
    check_quota(db, user)
    job = Job(owner_id=user.id, kind=kind, name=name or kind, spec=spec,
              backend=get_settings().launcher)
    db.add(job)
    db.commit()
    launch_job(db, job, user)
    return job_dict(job)


@router.post('/estimate')
def estimate_train(body: TrainIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    spec = _validate_train(db, body, user)
    db.rollback()  # validation bumps use_count; an estimate should not
    return estimate(spec)


@router.post('/train', status_code=201)
def create_train(body: TrainIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    spec = _validate_train(db, body, user)
    spec['estimate'] = estimate(spec)
    name = body.name or f'train {body.preset} {body.variant} s{body.seed}'
    return _create(db, user, 'train', name, spec)


@router.post('/evaluate', status_code=201)
def create_evaluate(body: EvaluateIn, user: User = Depends(current_user),
                    db: Session = Depends(get_db)):
    model = get_readable(db, WorldModel, body.model_id, user)
    if not ENV_PRESETS.get(model.env_preset, {}).get('simulator'):
        raise HTTPException(422,
                            'planning evaluation needs a built-in environment simulator')
    planner = body.planner or default_planner(model.num_levels)
    if planner not in planners_for(model.num_levels):
        raise HTTPException(422,
                            f'planner {planner!r} does not fit a {model.num_levels}-level model')
    spec = {'model_id': model.id, 'env': model.env_preset, 'variant': model.variant,
            'levels': model.num_levels, 'planner': planner, 'num_eval': body.num_eval,
            'seed': body.seed, 'gpu': _check_gpu(body.gpu), 'max_hours': _max_hours(body.max_hours)}
    model.use_count += 1
    return _create(db, user, 'evaluate', f'eval {model.name} [{planner}]', spec)


def _default_sim_dataset(db: Session, model: WorldModel, user: User) -> Dataset:
    if model.train_dataset_id:
        ds = db.get(Dataset, model.train_dataset_id)
        if ds is not None and ds.status == 'ready' and (ds.visibility == 'public' or ds.owner_id == user.id):
            return ds
    ds = db.scalar(select(Dataset).where(Dataset.source == 'official',
                                         Dataset.env_preset == model.env_preset))
    if ds is None:
        raise HTTPException(422,
                            'choose a dataset to take start frames and actions from')
    return ds


def validate_sim_target(db: Session, user: User, model_id: str, dataset_id: str | None,
                        levels: list[int] | None) -> tuple[WorldModel, Dataset, list[int]]:
    model = get_readable(db, WorldModel, model_id, user)
    ds = _ready_dataset(db, dataset_id, user) if dataset_id else _default_sim_dataset(db, model, user)
    if ds.source != 'official' and model.env_preset != 'custom' and ds.env_preset not in (model.env_preset, 'custom'):
        raise HTTPException(422,
                            f'dataset is {ds.env_preset!r} data but the model is {model.env_preset!r}')
    if ds.source == 'official' and ds.env_preset != model.env_preset:
        raise HTTPException(422, 'dataset and model environments differ')
    levels = sorted(set(levels or range(1, model.num_levels + 1)))
    if any(lv < 1 or lv > model.num_levels for lv in levels):
        raise HTTPException(422,
                            f'levels must be within 1..{model.num_levels}')
    return model, ds, levels


@router.post('/simulate', status_code=201)
def create_simulate(body: SimulateIn, user: User = Depends(current_user),
                    db: Session = Depends(get_db)):
    model, ds, levels = validate_sim_target(db, user, body.model_id, body.dataset_id, body.levels)
    spec = body.model_dump(exclude={'gpu', 'max_hours', 'levels', 'dataset_id'})
    spec.update(dataset_id=ds.id, levels=levels, env=model.env_preset, variant=model.variant,
                num_levels=model.num_levels, gpu=_check_gpu(body.gpu),
                max_hours=_max_hours(body.max_hours))
    model.use_count += 1
    return _create(db, user, 'simulate', f'dream {model.name}', spec)


@router.get('')
def list_jobs(kind: str | None = None, active: bool = False, limit: int = Query(100, le=500),
              user: User = Depends(current_user), db: Session = Depends(get_db)):
    stmt = select(Job).where(Job.owner_id == user.id)
    if kind:
        stmt = stmt.where(Job.kind == kind)
    if active:
        stmt = stmt.where(Job.status.in_(ACTIVE_JOB_STATES))
    return [job_dict(j) for j in db.scalars(stmt.order_by(Job.created_at.desc()).limit(limit))]


@router.get('/{job_id}')
def get_job(job_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    return job_dict(get_owned(db, Job, job_id, user))


@router.post('/{job_id}/cancel')
def cancel_job(job_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    job = get_owned(db, Job, job_id, user)
    if job.status not in ACTIVE_JOB_STATES:
        raise HTTPException(status.HTTP_409_CONFLICT, f'job is already {job.status}')
    job.cancel_requested = True
    db.commit()
    if job.status in ('queued', 'provisioning') or job.last_heartbeat is None:
        # The worker has not checked in yet: nothing to stop gracefully.
        finalize_job(db, job, 'cancelled', 'cancelled by user')
    return job_dict(job)


@router.get('/{job_id}/logs')
def job_logs(job_id: str, after: int = 0, limit: int = Query(500, le=5000),
             user: User = Depends(current_user), db: Session = Depends(get_db)):
    get_owned(db, Job, job_id, user)
    rows = db.scalars(select(JobLog).where(JobLog.job_id == job_id, JobLog.id > after)
                      .order_by(JobLog.id).limit(limit)).all()
    return {'lines': [{'id': r.id, 'ts': iso(r.ts), 'line': r.line} for r in rows],
            'last_id': rows[-1].id if rows else after}


@router.get('/{job_id}/metrics')
def job_metrics(job_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    get_owned(db, Job, job_id, user)
    series: dict[str, list] = {}
    for r in db.scalars(select(JobMetric).where(JobMetric.job_id == job_id).order_by(JobMetric.id)):
        series.setdefault(r.name, []).append([r.step, r.value])
    return series


@router.get('/{job_id}/artifacts')
def job_artifacts(job_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    get_owned(db, Job, job_id, user)
    rows = db.scalars(select(JobArtifact).where(JobArtifact.job_id == job_id,
                                                JobArtifact.size_bytes > 0)
                      .order_by(JobArtifact.id)).all()
    return [{'name': r.name, 'content_type': r.content_type, 'size_bytes': r.size_bytes,
             'url': signed_file_url(r.storage_key, ttl_s=6 * 3600)} for r in rows]
