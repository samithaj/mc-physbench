"""Interactive simulation sessions: a GPU worker keeps a world model loaded and the
browser steps it through this proxy (the worker itself is never exposed to users)."""

from __future__ import annotations

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db import get_db
from ..deps import current_user, get_owned
from ..models import SimSession, User, utcnow
from ..serializers import session_dict
from ..services import ACTIVE_SESSION_STATES, check_quota, launch_session, stop_session
from .jobs import GpuIn, _check_gpu, validate_sim_target

router = APIRouter(prefix='/api/sessions', tags=['sessions'])

ALLOWED_CALLS = {'info', 'reset', 'step', 'rollout', 'sample_action'}


class SessionIn(BaseModel):
    model_id: str
    dataset_id: str | None = None
    gpu: GpuIn = Field(default_factory=GpuIn)
    idle_minutes: int | None = Field(default=None, ge=5, le=240)


@router.post('', status_code=201)
def create_session(body: SessionIn, user: User = Depends(current_user),
                   db: Session = Depends(get_db)):
    model, ds, _ = validate_sim_target(db, user, body.model_id, body.dataset_id, None)
    check_quota(db, user)
    sess = SimSession(
        owner_id=user.id, model_id=model.id, dataset_id=ds.id,
        backend=get_settings().launcher,
        spec={'gpu': _check_gpu(body.gpu), 'env': model.env_preset, 'variant': model.variant,
              'num_levels': model.num_levels, 'model_name': model.name,
              'idle_minutes': body.idle_minutes or get_settings().session_idle_minutes},
    )
    db.add(sess)
    model.use_count += 1
    db.commit()
    launch_session(db, sess, user)
    return session_dict(sess)


@router.get('')
def list_sessions(user: User = Depends(current_user), db: Session = Depends(get_db)):
    rows = db.scalars(select(SimSession).where(SimSession.owner_id == user.id)
                      .order_by(SimSession.created_at.desc()).limit(50)).all()
    return [session_dict(s) for s in rows]


@router.get('/{sess_id}')
def get_session(sess_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    return session_dict(get_owned(db, SimSession, sess_id, user))


@router.post('/{sess_id}/stop')
def stop(sess_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    sess = get_owned(db, SimSession, sess_id, user)
    stop_session(db, sess, 'stopped', 'stopped by user')
    return session_dict(sess)


@router.post('/{sess_id}/call/{action}')
async def call(sess_id: str, action: str, payload: dict | None = None,
               user: User = Depends(current_user), db: Session = Depends(get_db)):
    sess = get_owned(db, SimSession, sess_id, user)
    if action not in ALLOWED_CALLS:
        raise HTTPException(status.HTTP_404_NOT_FOUND, 'unknown simulator call')
    if sess.status != 'ready' or not sess.endpoint_url:
        raise HTTPException(status.HTTP_409_CONFLICT, f'session is {sess.status}')
    sess.last_used_at = utcnow()
    db.commit()
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f'{sess.endpoint_url}/{action}', json=payload or {},
                                     headers={'X-Session-Secret': sess.worker_secret})
    except httpx.HTTPError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f'simulator unreachable: {exc}')
    if resp.status_code >= 400:
        detail = resp.json().get('error') if resp.headers.get('content-type', '').startswith(
            'application/json') else resp.text[:500]
        raise HTTPException(status.HTTP_502_BAD_GATEWAY if resp.status_code >= 500 else 400,
                            detail or 'simulator error')
    return resp.json()


def reap_idle_sessions(db: Session) -> None:
    now = utcnow()
    for sess in db.scalars(select(SimSession).where(SimSession.status.in_(ACTIVE_SESSION_STATES))):
        idle = (now - sess.last_used_at).total_seconds() / 60
        if sess.status == 'starting' and (now - sess.created_at).total_seconds() > 30 * 60:
            stop_session(db, sess, 'failed', 'worker did not become ready within 30 minutes')
        elif idle > sess.spec.get('idle_minutes', get_settings().session_idle_minutes):
            stop_session(db, sess, 'stopped', f'stopped after {int(idle)} idle minutes')
