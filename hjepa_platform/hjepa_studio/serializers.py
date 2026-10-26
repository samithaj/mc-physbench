"""ORM rows -> JSON-safe dicts for the API."""

from __future__ import annotations

from datetime import datetime

from .models import Dataset, Job, SimSession, User, WorldModel, utcnow
from .presets import ENV_PRESETS, VARIANTS
from .security import signed_file_url


def iso(dt: datetime | None) -> str | None:
    return dt.isoformat() + 'Z' if dt else None


def user_dict(user: User, private: bool = False) -> dict:
    out = {'id': user.id, 'username': user.username}
    if private:
        out.update(email=user.email, is_admin=user.is_admin,
                   has_runpod_key=bool(user.runpod_api_key_enc), created_at=iso(user.created_at))
    return out


def dataset_dict(ds: Dataset, viewer: User | None = None) -> dict:
    return {
        'id': ds.id,
        'name': ds.name,
        'description': ds.description,
        'visibility': ds.visibility,
        'source': ds.source,
        'source_ref': ds.source_ref,
        'env_preset': ds.env_preset,
        'env_label': ENV_PRESETS.get(ds.env_preset, {}).get('label', ds.env_preset),
        'status': ds.status,
        'size_bytes': ds.size_bytes,
        'expected_bytes': ds.expected_bytes,
        'inspection': ds.inspection,
        'preview_url': signed_file_url(ds.preview_key, ttl_s=6 * 3600) if ds.preview_key else None,
        'license': ds.license,
        'tags': ds.tags,
        'use_count': ds.use_count,
        'owner': user_dict(ds.owner) if ds.owner else None,
        'is_owner': bool(viewer and viewer.id == ds.owner_id),
        'created_at': iso(ds.created_at),
    }


def model_dict(m: WorldModel, viewer: User | None = None) -> dict:
    return {
        'id': m.id,
        'name': m.name,
        'description': m.description,
        'visibility': m.visibility,
        'env_preset': m.env_preset,
        'env_label': ENV_PRESETS.get(m.env_preset, {}).get('label', m.env_preset),
        'variant': m.variant,
        'variant_label': VARIANTS.get(m.variant, {}).get('label', m.variant),
        'num_levels': m.num_levels,
        'source': m.source,
        'source_ref': m.source_ref,
        'job_id': m.job_id,
        'train_dataset_id': m.train_dataset_id,
        'files': sorted(m.files.keys()),
        'has_decoders': any(k.startswith('decoder_level') for k in m.files)
                        or m.source == 'huggingface',
        'metrics': m.metrics,
        'use_count': m.use_count,
        'owner': user_dict(m.owner) if m.owner else None,
        'is_owner': bool(viewer and viewer.id == m.owner_id),
        'created_at': iso(m.created_at),
    }


def _elapsed_hours(start: datetime | None, end: datetime | None) -> float:
    if not start:
        return 0.0
    return max(0.0, ((end or utcnow()) - start).total_seconds() / 3600)


def job_dict(job: Job) -> dict:
    hours = _elapsed_hours(job.started_at or (job.created_at if job.pod_id else None),
                           job.finished_at)
    return {
        'id': job.id,
        'kind': job.kind,
        'name': job.name,
        'status': job.status,
        'spec': job.spec,
        'backend': job.backend,
        'pod_id': job.pod_id,
        'pod_info': job.pod_info,
        'progress': job.progress,
        'message': job.message,
        'result': job.result,
        'output_model_id': job.output_model_id,
        'cost_per_hr': job.cost_per_hr,
        'cost_usd': round(job.cost_per_hr * hours, 4),
        'runtime_hours': round(hours, 4),
        'cancel_requested': job.cancel_requested,
        'created_at': iso(job.created_at),
        'started_at': iso(job.started_at),
        'finished_at': iso(job.finished_at),
        'last_heartbeat': iso(job.last_heartbeat),
    }


def session_dict(s: SimSession) -> dict:
    hours = _elapsed_hours(s.created_at, s.stopped_at)
    return {
        'id': s.id,
        'model_id': s.model_id,
        'dataset_id': s.dataset_id,
        'status': s.status,
        'backend': s.backend,
        'pod_id': s.pod_id,
        'spec': s.spec,
        'info': s.info,
        'message': s.message,
        'cost_per_hr': s.cost_per_hr,
        'cost_usd': round(s.cost_per_hr * hours, 4),
        'created_at': iso(s.created_at),
        'last_used_at': iso(s.last_used_at),
        'stopped_at': iso(s.stopped_at),
    }
