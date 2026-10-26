from __future__ import annotations

import re

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db import get_db
from ..deps import current_user
from ..models import User
from ..security import create_access_token, encrypt_secret, hash_password, verify_password
from ..serializers import user_dict
from ..services import SYSTEM_USERNAME

router = APIRouter(prefix='/api/auth', tags=['auth'])

_USERNAME_RE = re.compile(r'^[a-zA-Z0-9_-]{3,32}$')
_EMAIL_RE = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')


class RegisterIn(BaseModel):
    email: str = Field(max_length=255)
    username: str
    password: str = Field(min_length=8, max_length=256)


class LoginIn(BaseModel):
    login: str  # email or username
    password: str


class SettingsIn(BaseModel):
    runpod_api_key: str | None = Field(default=None, max_length=256)
    clear_runpod_api_key: bool = False


def _token_response(user: User) -> dict:
    return {'access_token': create_access_token(user.id), 'token_type': 'bearer',
            'user': user_dict(user, private=True)}


@router.post('/register')
def register(body: RegisterIn, db: Session = Depends(get_db)):
    if not get_settings().allow_registration:
        raise HTTPException(status.HTTP_403_FORBIDDEN, 'registration is closed')
    email = body.email.strip().lower()
    if not _EMAIL_RE.match(email):
        raise HTTPException(422, 'invalid email')
    if not _USERNAME_RE.match(body.username) or body.username.lower() == SYSTEM_USERNAME:
        raise HTTPException(422,
                            'username must be 3-32 letters, digits, - or _')
    clash = db.scalar(select(User).where(or_(User.email == email,
                                             User.username == body.username)))
    if clash:
        raise HTTPException(status.HTTP_409_CONFLICT, 'email or username already registered')
    # The first real account administers the instance.
    first = db.scalar(select(User).where(User.is_system.is_(False))) is None
    user = User(email=email, username=body.username,
                password_hash=hash_password(body.password), is_admin=first)
    db.add(user)
    db.commit()
    return _token_response(user)


@router.post('/login')
def login(body: LoginIn, db: Session = Depends(get_db)):
    ident = body.login.strip()
    user = db.scalar(select(User).where(or_(User.email == ident.lower(), User.username == ident)))
    if user is None or user.is_system or not verify_password(body.password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, 'wrong login or password')
    return _token_response(user)


@router.get('/me')
def me(user: User = Depends(current_user)):
    return user_dict(user, private=True)


@router.put('/settings')
def update_settings(body: SettingsIn, user: User = Depends(current_user),
                    db: Session = Depends(get_db)):
    if body.clear_runpod_api_key:
        user.runpod_api_key_enc = None
    elif body.runpod_api_key:
        key = body.runpod_api_key.strip()
        if not re.fullmatch(r'[A-Za-z0-9_\-]{16,256}', key):
            raise HTTPException(422, 'that does not look like a RunPod API key')
        user.runpod_api_key_enc = encrypt_secret(key)
    db.commit()
    return user_dict(user, private=True)
