from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from ..db import get_db
from ..deps import current_user, get_owned, get_readable
from ..models import User, WorldModel
from ..security import signed_file_url
from ..serializers import model_dict

router = APIRouter(prefix='/api/models', tags=['models'])


class PatchModel(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=128)
    description: str | None = Field(default=None, max_length=5000)
    visibility: str | None = Field(default=None, pattern='^(private|public)$')


@router.get('')
def list_models(scope: str = Query('all', pattern='^(mine|public|all)$'),
                env: str | None = None, variant: str | None = None, q: str | None = None,
                user: User = Depends(current_user), db: Session = Depends(get_db)):
    stmt = select(WorldModel)
    if scope == 'mine':
        stmt = stmt.where(WorldModel.owner_id == user.id)
    elif scope == 'public':
        stmt = stmt.where(WorldModel.visibility == 'public')
    else:
        stmt = stmt.where(or_(WorldModel.owner_id == user.id, WorldModel.visibility == 'public'))
    if env:
        stmt = stmt.where(WorldModel.env_preset == env)
    if variant:
        stmt = stmt.where(WorldModel.variant == variant)
    if q:
        stmt = stmt.where(WorldModel.name.ilike(f'%{q}%'))
    rows = db.scalars(stmt.order_by(WorldModel.created_at.desc()).limit(300)).all()
    return [model_dict(m, user) for m in rows]


@router.get('/leaderboard')
def leaderboard(env: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Visible models for one env ranked by their best planning success rate."""
    rows = db.scalars(select(WorldModel).where(
        WorldModel.env_preset == env,
        or_(WorldModel.owner_id == user.id, WorldModel.visibility == 'public'))).all()
    board = []
    for m in rows:
        planning = (m.metrics or {}).get('planning') or {}
        scored = [(p, v.get('success_rate')) for p, v in planning.items()
                  if isinstance(v, dict) and v.get('success_rate') is not None]
        if not scored:
            continue
        planner, best = max(scored, key=lambda pv: pv[1])
        board.append({'model': model_dict(m, user), 'planner': planner, 'success_rate': best})
    board.sort(key=lambda r: r['success_rate'], reverse=True)
    return board


@router.get('/{model_id}')
def get_model(model_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    return model_dict(get_readable(db, WorldModel, model_id, user), user)


@router.patch('/{model_id}')
def patch_model(model_id: str, body: PatchModel, user: User = Depends(current_user),
                db: Session = Depends(get_db)):
    m = get_owned(db, WorldModel, model_id, user)
    for field in ('name', 'description', 'visibility'):
        value = getattr(body, field)
        if value is not None:
            setattr(m, field, value)
    db.commit()
    return model_dict(m, user)


@router.delete('/{model_id}', status_code=204)
def delete_model(model_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    m = get_owned(db, WorldModel, model_id, user)
    db.delete(m)  # files stay with the job that produced them
    db.commit()


@router.get('/{model_id}/files/{name}')
def model_file(model_id: str, name: str, user: User = Depends(current_user),
               db: Session = Depends(get_db)):
    m = get_readable(db, WorldModel, model_id, user)
    if m.source == 'huggingface':
        ref = m.source_ref
        return {'url': f'https://huggingface.co/{ref["repo"]}/tree/main/{ref["prefix"]}'}
    key = m.files.get(name)
    if key is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, 'no such file')
    m.use_count += 1
    db.commit()
    return {'url': signed_file_url(key, ttl_s=3600)}
