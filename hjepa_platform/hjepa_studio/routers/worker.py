"""Endpoints GPU workers call back into, authenticated by a per-job/session token."""

from __future__ import annotations

import re

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db import get_db
from ..deps import worker_token
from ..models import Job, JobArtifact, JobLog, JobMetric, SimSession, WorldModel, Dataset, utcnow
from ..security import signed_file_url, tokens_match
from ..services import (ACTIVE_SESSION_STATES, TERMINAL_JOB_STATES, dataset_worker_input,
                        finalize_job, model_worker_input, worker_spec_for_job)

router = APIRouter(prefix='/api/worker', tags=['worker'])

MAX_LOG_LINES_PER_JOB = 50_000
MAX_WORKER_ATTEMPTS = 3
_ARTIFACT_NAME_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,200}$')


def _job_for(db: Session, job_id: str, token: str) -> Job:
    job = db.get(Job, job_id)
    if job is None or not tokens_match(token, job.token_hash):
        raise HTTPException(status.HTTP_403_FORBIDDEN, 'bad worker token')
    return job


def _session_for(db: Session, sess_id: str, token: str) -> SimSession:
    sess = db.get(SimSession, sess_id)
    if sess is None or not tokens_match(token, sess.token_hash):
        raise HTTPException(status.HTTP_403_FORBIDDEN, 'bad worker token')
    return sess


class HeartbeatIn(BaseModel):
    status: str | None = Field(default=None, pattern='^(running|uploading)$')
    progress: float | None = Field(default=None, ge=0, le=1)
    message: str | None = Field(default=None, max_length=2000)


class LogsIn(BaseModel):
    lines: list[str] = Field(max_length=2000)


class MetricRow(BaseModel):
    name: str = Field(max_length=128)
    value: float
    step: int = 0
    epoch: int | None = None


class MetricsIn(BaseModel):
    rows: list[MetricRow] = Field(max_length=5000)


class ArtifactIn(BaseModel):
    name: str
    content_type: str = Field(default='application/octet-stream', max_length=128)


class CompleteIn(BaseModel):
    status: str = Field(pattern='^(succeeded|failed|cancelled)$')
    message: str = Field(default='', max_length=5000)
    result: dict = Field(default_factory=dict)


@router.get('/jobs/{job_id}/spec')
def job_spec(job_id: str, token: str = Depends(worker_token), db: Session = Depends(get_db)):
    job = _job_for(db, job_id, token)
    if job.status in TERMINAL_JOB_STATES:
        raise HTTPException(status.HTTP_409_CONFLICT, f'job is {job.status}')
    # A pod restarts its container when the worker crashes; don't loop until the time limit.
    attempts = int((job.pod_info or {}).get('attempts', 0)) + 1
    job.pod_info = {**(job.pod_info or {}), 'attempts': attempts}
    if attempts > MAX_WORKER_ATTEMPTS:
        finalize_job(db, job, 'failed', f'worker crashed {attempts - 1} times; see logs')
        raise HTTPException(status.HTTP_409_CONFLICT, 'job failed')
    if job.status in ('queued', 'provisioning'):
        job.status = 'running'
        job.started_at = job.started_at or utcnow()
    job.last_heartbeat = utcnow()
    job.message = 'worker started'
    db.commit()
    return worker_spec_for_job(db, job)


@router.post('/jobs/{job_id}/heartbeat')
def job_heartbeat(job_id: str, body: HeartbeatIn, token: str = Depends(worker_token),
                  db: Session = Depends(get_db)):
    job = _job_for(db, job_id, token)
    if job.status in TERMINAL_JOB_STATES:
        return {'cancel': True}
    job.last_heartbeat = utcnow()
    if body.status:
        job.status = body.status
    if body.progress is not None:
        job.progress = body.progress
    if body.message is not None:
        job.message = body.message
    db.commit()
    return {'cancel': job.cancel_requested}


@router.post('/jobs/{job_id}/logs')
def job_logs(job_id: str, body: LogsIn, token: str = Depends(worker_token),
             db: Session = Depends(get_db)):
    job = _job_for(db, job_id, token)
    count = db.scalar(select(func.count()).select_from(JobLog).where(JobLog.job_id == job.id))
    room = max(0, MAX_LOG_LINES_PER_JOB - count)
    db.add_all(JobLog(job_id=job.id, line=line[:4000]) for line in body.lines[:room])
    job.last_heartbeat = utcnow()
    db.commit()
    return {'accepted': min(room, len(body.lines))}


@router.post('/jobs/{job_id}/metrics')
def job_metrics(job_id: str, body: MetricsIn, token: str = Depends(worker_token),
                db: Session = Depends(get_db)):
    job = _job_for(db, job_id, token)
    db.add_all(JobMetric(job_id=job.id, name=r.name, value=r.value, step=r.step, epoch=r.epoch)
               for r in body.rows)
    db.commit()
    return {'accepted': len(body.rows)}


@router.post('/jobs/{job_id}/artifacts')
def job_artifact_upload_url(job_id: str, body: ArtifactIn, token: str = Depends(worker_token),
                            db: Session = Depends(get_db)):
    job = _job_for(db, job_id, token)
    if job.status in TERMINAL_JOB_STATES:
        raise HTTPException(status.HTTP_409_CONFLICT, f'job is {job.status}')
    if not _ARTIFACT_NAME_RE.match(body.name):
        raise HTTPException(422, 'bad artifact name')
    key = f'jobs/{job.id}/{body.name}'
    row = db.scalar(select(JobArtifact).where(JobArtifact.job_id == job.id,
                                              JobArtifact.name == body.name))
    if row is None:
        db.add(JobArtifact(job_id=job.id, name=body.name, storage_key=key,
                           content_type=body.content_type))
    db.commit()
    return {'key': key, 'upload_url': signed_file_url(key, method='PUT', ttl_s=6 * 3600,
                                                      absolute=True)}


@router.post('/jobs/{job_id}/complete')
def job_complete(job_id: str, body: CompleteIn, token: str = Depends(worker_token),
                 db: Session = Depends(get_db)):
    job = _job_for(db, job_id, token)
    final = 'cancelled' if job.cancel_requested and body.status != 'succeeded' else body.status
    finalize_job(db, job, final, body.message, body.result)
    return {'status': job.status, 'output_model_id': job.output_model_id}


# --------------------------------------------------------------------------- sessions

@router.get('/sessions/{sess_id}/spec')
def session_spec(sess_id: str, token: str = Depends(worker_token), db: Session = Depends(get_db)):
    sess = _session_for(db, sess_id, token)
    if sess.status not in ACTIVE_SESSION_STATES:
        raise HTTPException(status.HTTP_409_CONFLICT, f'session is {sess.status}')
    model = db.get(WorldModel, sess.model_id)
    ds = db.get(Dataset, sess.dataset_id) if sess.dataset_id else None
    if model is None:
        raise HTTPException(status.HTTP_410_GONE, 'model was deleted')
    from ..presets import HF_REPO, HJEPA_COMMIT, HJEPA_REPO
    return {
        'session_id': sess.id,
        'worker_secret': sess.worker_secret,
        'hjepa': {'repo': HJEPA_REPO, 'commit': HJEPA_COMMIT, 'hf_repo': HF_REPO},
        'params': sess.spec,
        'inputs': {'model': model_worker_input(model),
                   'datasets': [dataset_worker_input(ds, 'sim')] if ds else []},
    }


class SessionReadyIn(BaseModel):
    info: dict = Field(default_factory=dict)
    failed: bool = False
    message: str = Field(default='', max_length=5000)


@router.post('/sessions/{sess_id}/ready')
def session_ready(sess_id: str, body: SessionReadyIn, token: str = Depends(worker_token),
                  db: Session = Depends(get_db)):
    sess = _session_for(db, sess_id, token)
    if sess.status not in ACTIVE_SESSION_STATES:
        return {'stop': True}
    if body.failed:
        from ..services import stop_session
        stop_session(db, sess, 'failed', body.message or 'worker failed to load the model')
        return {'stop': True}
    sess.status = 'ready'
    sess.info = {**(sess.info or {}), **body.info}
    sess.message = body.message or 'ready'
    sess.last_used_at = utcnow()
    db.commit()
    return {'stop': False}


@router.post('/sessions/{sess_id}/heartbeat')
def session_heartbeat(sess_id: str, token: str = Depends(worker_token),
                      db: Session = Depends(get_db)):
    sess = _session_for(db, sess_id, token)
    return {'stop': sess.status not in ACTIVE_SESSION_STATES}
