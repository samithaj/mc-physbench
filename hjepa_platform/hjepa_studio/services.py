"""Job/session lifecycle shared by the API routers and the background reaper."""

from __future__ import annotations

import logging
import secrets

from fastapi import HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .config import get_settings
from .launchers import LaunchError, get_launcher
from .models import (Dataset, Job, JobArtifact, SimSession, User, WorldModel, utcnow)
from .presets import ENV_PRESETS, HF_REPO, HJEPA_COMMIT, HJEPA_REPO, VARIANTS, official_model_catalog
from .security import decrypt_secret, hash_password, new_worker_token, signed_file_url

log = logging.getLogger(__name__)

ACTIVE_JOB_STATES = ('queued', 'provisioning', 'running', 'uploading')
TERMINAL_JOB_STATES = ('succeeded', 'failed', 'cancelled')
ACTIVE_SESSION_STATES = ('starting', 'ready')
WORKER_URL_TTL = 24 * 3600  # signed input URLs must outlive slow downloads


# --------------------------------------------------------------------------- keys / quota

def runpod_key_for(user: User) -> tuple[str | None, str]:
    """Return (api_key, payer). A user's own key wins over the platform key."""
    own = decrypt_secret(user.runpod_api_key_enc)
    if own:
        return own, 'user'
    return get_settings().runpod_api_key, 'platform'


def _key_for_payer(db: Session, owner_id: str, payer: str | None) -> str | None:
    if payer == 'user':
        user = db.get(User, owner_id)
        own = decrypt_secret(user.runpod_api_key_enc) if user else None
        if own:
            return own
    return get_settings().runpod_api_key


def check_quota(db: Session, user: User) -> None:
    limit = get_settings().max_concurrent_jobs
    if user.is_admin:
        return
    jobs = db.scalar(select(func.count()).select_from(Job).where(
        Job.owner_id == user.id, Job.status.in_(ACTIVE_JOB_STATES)))
    sessions = db.scalar(select(func.count()).select_from(SimSession).where(
        SimSession.owner_id == user.id, SimSession.status.in_(ACTIVE_SESSION_STATES)))
    if jobs + sessions >= limit:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS,
                            f'limit of {limit} concurrent GPU jobs/sessions reached')


def _require_backend_ready(user: User) -> None:
    settings = get_settings()
    if settings.launcher == 'runpod' and not runpod_key_for(user)[0]:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            'no RunPod API key: add yours in Settings or ask the admin to '
                            'configure RUNPOD_API_KEY')


# --------------------------------------------------------------------------- jobs

def launch_job(db: Session, job: Job, user: User) -> Job:
    _require_backend_ready(user)
    token, digest = new_worker_token()
    api_key, payer = runpod_key_for(user)
    job.token_hash = digest
    job.status = 'provisioning'
    job.message = 'requesting GPU'
    db.commit()
    launcher = get_launcher()
    try:
        res = launcher.launch(mode='job', ident=job.id, token=token, gpu=job.spec['gpu'],
                              api_key=api_key)
    except LaunchError as exc:
        job.status = 'failed'
        job.message = f'could not start GPU worker: {exc}'
        job.finished_at = utcnow()
        db.commit()
        return job
    job.backend = res.backend
    job.pod_id = res.pod_id
    job.cost_per_hr = res.cost_per_hr
    job.pod_info = {**res.info, 'payer': payer}
    job.message = 'GPU pod starting, pulling worker image'
    db.commit()
    return job


def terminate_job_pod(db: Session, job: Job) -> None:
    if not job.pod_id:
        return
    try:
        get_launcher(job.backend).terminate(
            job.pod_id, _key_for_payer(db, job.owner_id, job.pod_info.get('payer')))
    except LaunchError as exc:
        log.warning('terminate failed for job %s: %s', job.id, exc)


def finalize_job(db: Session, job: Job, final_status: str, message: str = '',
                 result: dict | None = None) -> Job:
    if job.status in TERMINAL_JOB_STATES:
        return job
    job.status = final_status
    job.finished_at = utcnow()
    if message:
        job.message = message
    if result:
        job.result = {**(job.result or {}), **result}
    if final_status == 'succeeded':
        job.progress = 1.0
        try:
            _apply_job_outputs(db, job)
        except Exception as exc:  # never leave the pod running because of a bookkeeping bug
            log.exception('applying outputs for job %s failed', job.id)
            job.status = 'failed'
            job.message = f'job finished but its outputs could not be registered: {exc}'
    db.commit()
    terminate_job_pod(db, job)
    return job


def _artifact_keys(db: Session, job: Job) -> dict[str, str]:
    rows = db.scalars(select(JobArtifact).where(JobArtifact.job_id == job.id)).all()
    return {r.name: r.storage_key for r in rows if r.size_bytes > 0}


def _apply_job_outputs(db: Session, job: Job) -> None:
    result = job.result or {}
    if job.kind == 'train':
        artifacts = _artifact_keys(db, job)
        files = {logical: artifacts[name]
                 for logical, name in (result.get('model_files') or {}).items()
                 if name in artifacts}
        if 'object_ckpt' not in files:
            raise ValueError('worker did not upload the model checkpoint')
        spec = job.spec
        model = WorldModel(
            owner_id=job.owner_id,
            name=spec.get('model_name') or job.name or f'{spec["preset"]}_{spec["variant"]}',
            description=f'Trained by job {job.id}: {spec["epochs"]} epochs, seed {spec["seed"]}.',
            visibility=spec.get('output_visibility', 'private'),
            env_preset=spec['preset'],
            variant=spec['variant'],
            num_levels=VARIANTS[spec['variant']]['levels'],
            source='job',
            job_id=job.id,
            train_dataset_id=spec.get('train_dataset_id'),
            files=files,
            metrics=result.get('metrics') or {},
        )
        db.add(model)
        db.flush()
        job.output_model_id = model.id
    elif job.kind == 'evaluate':
        model = db.get(WorldModel, job.spec['model_id'])
        if model is not None and result.get('metrics'):
            planning = dict((model.metrics or {}).get('planning') or {})
            planning[job.spec['planner']] = {**result['metrics'], 'job_id': job.id}
            model.metrics = {**(model.metrics or {}), 'planning': planning}


def dataset_worker_input(ds: Dataset, role: str) -> dict:
    if ds.source == 'official':
        return {'role': role, 'source': 'official', 'preset': ds.source_ref['preset']}
    if ds.source == 'huggingface':
        return {'role': role, 'source': 'huggingface', 'name': f'ds_{ds.id}', **ds.source_ref}
    return {'role': role, 'source': 'url', 'name': f'ds_{ds.id}',
            'url': signed_file_url(ds.storage_key, ttl_s=WORKER_URL_TTL, absolute=True),
            'size_bytes': ds.size_bytes}


def model_worker_input(m: WorldModel) -> dict:
    if m.source == 'huggingface':
        return {'source': 'huggingface', **m.source_ref,
                'env': m.env_preset, 'variant': m.variant, 'levels': m.num_levels}
    return {'source': 'url', 'env': m.env_preset, 'variant': m.variant, 'levels': m.num_levels,
            'files': {name: signed_file_url(key, ttl_s=WORKER_URL_TTL, absolute=True)
                      for name, key in m.files.items()}}


def worker_spec_for_job(db: Session, job: Job) -> dict:
    spec = job.spec
    inputs: dict = {'datasets': []}
    for role in ('train', 'val'):
        ds_id = spec.get(f'{role}_dataset_id')
        if ds_id:
            inputs['datasets'].append(dataset_worker_input(db.get(Dataset, ds_id), role))
    if spec.get('dataset_id'):
        inputs['datasets'].append(dataset_worker_input(db.get(Dataset, spec['dataset_id']), 'sim'))
    if spec.get('model_id'):
        inputs['model'] = model_worker_input(db.get(WorldModel, spec['model_id']))
    params = {k: v for k, v in spec.items() if k not in ('gpu',)}
    return {
        'job_id': job.id,
        'kind': job.kind,
        'hjepa': {'repo': HJEPA_REPO, 'commit': HJEPA_COMMIT, 'hf_repo': HF_REPO},
        'params': params,
        'gpu': spec.get('gpu', {}),
        'inputs': inputs,
    }


# --------------------------------------------------------------------------- sessions

def launch_session(db: Session, sess: SimSession, user: User) -> SimSession:
    _require_backend_ready(user)
    token, digest = new_worker_token()
    api_key, payer = runpod_key_for(user)
    sess.token_hash = digest
    sess.worker_secret = secrets.token_urlsafe(32)
    db.commit()
    try:
        res = get_launcher().launch(mode='serve', ident=sess.id, token=token,
                                    gpu=sess.spec['gpu'], api_key=api_key)
    except LaunchError as exc:
        sess.status = 'failed'
        sess.message = f'could not start GPU worker: {exc}'
        sess.stopped_at = utcnow()
        db.commit()
        return sess
    sess.backend = res.backend
    sess.pod_id = res.pod_id
    sess.endpoint_url = res.endpoint_url
    sess.cost_per_hr = res.cost_per_hr
    sess.info = {**res.info, 'payer': payer}
    sess.message = 'GPU pod starting, loading model'
    db.commit()
    return sess


def stop_session(db: Session, sess: SimSession, final_status: str = 'stopped',
                 message: str = '') -> SimSession:
    if sess.status in ('stopped', 'failed'):
        return sess
    sess.status = final_status
    sess.stopped_at = utcnow()
    if message:
        sess.message = message
    db.commit()
    if sess.pod_id:
        get_launcher(sess.backend).terminate(
            sess.pod_id, _key_for_payer(db, sess.owner_id, (sess.info or {}).get('payer')))
    return sess


# --------------------------------------------------------------------------- catalog seed

SYSTEM_USERNAME = 'h-jepa'


def seed_catalog(db: Session) -> None:
    """Expose the official H-JEPA datasets and released checkpoints as public entries."""
    system = db.scalar(select(User).where(User.username == SYSTEM_USERNAME))
    if system is None:
        system = User(email='official@h-jepa.invalid', username=SYSTEM_USERNAME,
                      password_hash=hash_password(secrets.token_urlsafe(32)), is_system=True)
        db.add(system)
        db.flush()

    existing_ds = {d.source_ref.get('preset') for d in db.scalars(
        select(Dataset).where(Dataset.owner_id == system.id, Dataset.source == 'official'))}
    for key, preset in ENV_PRESETS.items():
        if not preset['official'] or key in existing_ds:
            continue
        official = preset['official']
        db.add(Dataset(
            owner_id=system.id,
            name=f'{preset["label"]} (official H-JEPA data)',
            description=(f'{preset["description"]} Train split `{official["train"]}`, '
                         f'val split `{official["val"]}`. Source: {official["source"]}. '
                         'Downloaded directly onto the GPU worker.'),
            visibility='public', source='official', env_preset=key,
            source_ref={'preset': key, **official}, status='ready',
            inspection={'valid': True, 'img_size': preset['img_size'],
                        'action_dim': preset['action_dim'], 'compatible_presets': [key]},
            license='MIT', tags=['official', key],
        ))

    existing_models = {m.source_ref.get('prefix') for m in db.scalars(
        select(WorldModel).where(WorldModel.owner_id == system.id))}
    for entry in official_model_catalog():
        if entry['hf_prefix'] in existing_models:
            continue
        db.add(WorldModel(
            owner_id=system.id, name=entry['name'],
            description='Released checkpoint from the H-JEPA paper (arXiv:2610.06805).',
            visibility='public', env_preset=entry['env'], variant=entry['variant'],
            num_levels=entry['levels'], source='huggingface',
            source_ref={'repo': HF_REPO, 'prefix': entry['hf_prefix'],
                        'object_ckpt': entry['object_ckpt']},
        ))
    db.commit()
