"""Password hashing, user JWTs, worker tokens, signed file URLs, secret encryption."""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import jwt
from cryptography.fernet import Fernet, InvalidToken

from .config import get_settings

_SCRYPT = dict(n=2**14, r=8, p=1, dklen=32)
ACCESS_TOKEN_TTL = timedelta(days=7)


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, **_SCRYPT)
    return 'scrypt$' + base64.b64encode(salt).decode() + '$' + base64.b64encode(digest).decode()


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, salt_b64, digest_b64 = stored.split('$')
    except ValueError:
        return False
    if scheme != 'scrypt':
        return False
    digest = hashlib.scrypt(password.encode(), salt=base64.b64decode(salt_b64), **_SCRYPT)
    return hmac.compare_digest(digest, base64.b64decode(digest_b64))


def create_access_token(user_id: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {'sub': user_id, 'iat': now, 'exp': now + ACCESS_TOKEN_TTL, 'typ': 'user'}
    return jwt.encode(payload, get_settings().secret_key, algorithm='HS256')


def decode_access_token(token: str) -> str | None:
    try:
        payload = jwt.decode(token, get_settings().secret_key, algorithms=['HS256'])
    except jwt.PyJWTError:
        return None
    if payload.get('typ') != 'user':
        return None
    return payload.get('sub')


def new_worker_token() -> tuple[str, str]:
    """Return (plaintext, sha256) for a per-job or per-session worker credential."""
    token = secrets.token_urlsafe(32)
    return token, token_hash(token)


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def tokens_match(token: str, stored_hash: str) -> bool:
    return bool(stored_hash) and hmac.compare_digest(token_hash(token), stored_hash)


def _url_signature(method: str, key: str, exp: int) -> str:
    msg = f'{method.upper()}\n{key}\n{exp}'.encode()
    return hmac.new(get_settings().secret_key.encode(), msg, hashlib.sha256).hexdigest()


def signed_file_url(key: str, method: str = 'GET', ttl_s: int = 3600, absolute: bool = False) -> str:
    exp = int(time.time()) + ttl_s
    sig = _url_signature(method, key, exp)
    path = f'/api/files/{quote(key)}?exp={exp}&sig={sig}'
    return (get_settings().public_url + path) if absolute else path


def verify_file_signature(method: str, key: str, exp: int, sig: str) -> bool:
    if exp < time.time():
        return False
    return hmac.compare_digest(_url_signature(method, key, exp), sig)


def _fernet() -> Fernet:
    raw = hashlib.sha256((get_settings().secret_key + ':fernet').encode()).digest()
    return Fernet(base64.urlsafe_b64encode(raw))


def encrypt_secret(value: str) -> str:
    return _fernet().encrypt(value.encode()).decode()


def decrypt_secret(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return _fernet().decrypt(value.encode()).decode()
    except InvalidToken:
        return None
