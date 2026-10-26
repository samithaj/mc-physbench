from __future__ import annotations

import time

from fastapi import APIRouter, Depends

from ..config import get_settings
from ..deps import current_user
from ..models import User
from ..presets import (ENV_PRESETS, GPU_CATALOG, HJEPA_COMMIT, HJEPA_REPO, VARIANTS, planners_for)
from ..runpod_client import RunPodClient, RunPodError
from ..services import runpod_key_for

router = APIRouter(prefix='/api/meta', tags=['meta'])

_price_cache: dict[str, tuple[float, list]] = {}
PRICE_TTL_S = 600


@router.get('/catalog')
def catalog():
    settings = get_settings()
    return {
        'presets': ENV_PRESETS,
        'variants': {k: {**v, 'planners': planners_for(v['levels'])} for k, v in VARIANTS.items()},
        'hjepa': {'repo': HJEPA_REPO, 'commit': HJEPA_COMMIT},
        'launcher': settings.launcher,
        'allow_registration': settings.allow_registration,
        'limits': {'max_hours': settings.max_hours_limit,
                   'max_concurrent_jobs': settings.max_concurrent_jobs},
    }


@router.get('/gpus')
def gpus(user: User = Depends(current_user)):
    """The small-GPU catalog, with live RunPod prices when an API key is available."""
    key, _ = runpod_key_for(user)
    live: dict[str, dict] = {}
    if key and get_settings().launcher == 'runpod':
        cached = _price_cache.get(key)
        if cached and time.time() - cached[0] < PRICE_TTL_S:
            rows = cached[1]
        else:
            try:
                rows = RunPodClient(key, timeout=10).gpu_types()
                _price_cache[key] = (time.time(), rows)
            except RunPodError:
                rows = []
        live = {r['id']: r for r in rows}
    out = []
    for g in GPU_CATALOG:
        row = live.get(g['id'])
        entry = {**g, 'live': bool(row)}
        if row:
            entry.update(secure_usd_hr=row.get('securePrice'),
                         community_usd_hr=row.get('communityPrice'),
                         stock=(row.get('lowestPrice') or {}).get('stockStatus'))
        out.append(entry)
    return out
