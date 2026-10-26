from __future__ import annotations

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from .db import get_db
from .models import Dataset, User, WorldModel
from .security import decode_access_token


def _bearer(authorization: str | None) -> str | None:
    if authorization and authorization.lower().startswith('bearer '):
        return authorization[7:].strip()
    return None


def current_user(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> User:
    token = _bearer(authorization)
    user_id = decode_access_token(token) if token else None
    user = db.get(User, user_id) if user_id else None
    if user is None or user.is_system:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, 'not authenticated',
                            headers={'WWW-Authenticate': 'Bearer'})
    return user


def worker_token(authorization: str | None = Header(default=None)) -> str:
    token = _bearer(authorization)
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, 'missing worker token')
    return token


def can_read(obj: Dataset | WorldModel, user: User) -> bool:
    return obj.visibility == 'public' or obj.owner_id == user.id or user.is_admin


def can_write(obj, user: User) -> bool:
    return obj.owner_id == user.id or user.is_admin


def get_readable(db: Session, cls, obj_id: str, user: User):
    obj = db.get(cls, obj_id)
    if obj is None or not can_read(obj, user):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f'{cls.__name__} not found')
    return obj


def get_owned(db: Session, cls, obj_id: str, user: User):
    obj = db.get(cls, obj_id)
    if obj is None or not (obj.owner_id == user.id or user.is_admin):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f'{cls.__name__} not found')
    return obj
