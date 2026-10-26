"""Background guard so no rented GPU outlives its job: time limits, lost workers,
vanished pods and idle simulation sessions."""

from __future__ import annotations

import asyncio
import logging

from sqlalchemy import select

from .config import get_settings
from .db import session_factory
from .launchers import get_launcher
from .models import Job, utcnow
from .routers.sessions import reap_idle_sessions
from .services import ACTIVE_JOB_STATES, _key_for_payer, finalize_job

log = logging.getLogger(__name__)

POD_CHECK_EVERY = 4  # reaper ticks between RunPod status polls


def reap_once(tick: int = 0) -> None:
    settings = get_settings()
    db = session_factory()()
    try:
        now = utcnow()
        for job in db.scalars(select(Job).where(Job.status.in_(ACTIVE_JOB_STATES))).all():
            since_start = (now - (job.started_at or job.created_at)).total_seconds()
            max_hours = float(job.spec.get('max_hours', settings.default_max_hours))
            if since_start > max_hours * 3600:
                finalize_job(db, job, 'failed', f'stopped at the {max_hours:g} h time limit')
                continue
            if job.last_heartbeat is None:
                # Image pulls on a cold RunPod host can take a while.
                if (now - job.created_at).total_seconds() > max(settings.heartbeat_timeout_s, 1800):
                    finalize_job(db, job, 'failed', 'worker never checked in')
                continue
            if (now - job.last_heartbeat).total_seconds() > settings.heartbeat_timeout_s:
                finalize_job(db, job, 'failed', 'lost contact with the worker')
                continue
            if job.pod_id and tick % POD_CHECK_EVERY == 0:
                alive = get_launcher(job.backend).is_alive(
                    job.pod_id, _key_for_payer(db, job.owner_id, job.pod_info.get('payer')))
                if alive is False:
                    finalize_job(db, job, 'failed', 'GPU pod exited unexpectedly')
        reap_idle_sessions(db)
    except Exception:
        log.exception('reaper tick failed')
    finally:
        db.close()


async def reaper_loop() -> None:
    interval = get_settings().reaper_interval_s
    tick = 0
    while True:
        await asyncio.to_thread(reap_once, tick)
        tick += 1
        await asyncio.sleep(interval)
