"""Database tables: users, shared datasets and world models, GPU jobs, sim sessions."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, BigInteger, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def new_id() -> str:
    return uuid.uuid4().hex[:16]


class User(Base):
    __tablename__ = 'users'

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    is_system: Mapped[bool] = mapped_column(Boolean, default=False)
    # Fernet-encrypted personal RunPod key; jobs bill to it when set.
    runpod_api_key_enc: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Dataset(Base):
    """One HDF5 file in H-JEPA's layout (ep_len, ep_offset, pixels, action, ...)."""

    __tablename__ = 'datasets'

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    owner_id: Mapped[str] = mapped_column(ForeignKey('users.id', ondelete='CASCADE'), index=True)
    name: Mapped[str] = mapped_column(String(128))
    description: Mapped[str] = mapped_column(Text, default='')
    visibility: Mapped[str] = mapped_column(String(16), default='private')  # private | public
    # upload | official | huggingface
    source: Mapped[str] = mapped_column(String(16), default='upload')
    env_preset: Mapped[str] = mapped_column(String(32), default='custom')
    # upload: storage key. official: preset key. huggingface: {repo, filename, repo_type}
    storage_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    source_ref: Mapped[dict] = mapped_column(JSON, default=dict)
    # uploading | processing | ready | invalid
    status: Mapped[str] = mapped_column(String(16), default='uploading')
    size_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    expected_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    inspection: Mapped[dict] = mapped_column(JSON, default=dict)
    preview_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    license: Mapped[str] = mapped_column(String(64), default='')
    tags: Mapped[list] = mapped_column(JSON, default=list)
    use_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    owner: Mapped[User] = relationship()


class WorldModel(Base):
    """A trained H-JEPA checkpoint plus its normalizer, config and optional decoders."""

    __tablename__ = 'world_models'

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    owner_id: Mapped[str] = mapped_column(ForeignKey('users.id', ondelete='CASCADE'), index=True)
    name: Mapped[str] = mapped_column(String(128))
    description: Mapped[str] = mapped_column(Text, default='')
    visibility: Mapped[str] = mapped_column(String(16), default='private')
    env_preset: Mapped[str] = mapped_column(String(32))
    variant: Mapped[str] = mapped_column(String(32))
    num_levels: Mapped[int] = mapped_column(Integer, default=1)
    source: Mapped[str] = mapped_column(String(16), default='job')  # job | huggingface
    source_ref: Mapped[dict] = mapped_column(JSON, default=dict)
    job_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    train_dataset_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # logical name -> storage key (object_ckpt, normalizer, config, decoder_level{N}, ...)
    files: Mapped[dict] = mapped_column(JSON, default=dict)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    use_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    owner: Mapped[User] = relationship()


class Job(Base):
    __tablename__ = 'jobs'

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    owner_id: Mapped[str] = mapped_column(ForeignKey('users.id', ondelete='CASCADE'), index=True)
    kind: Mapped[str] = mapped_column(String(16))  # train | evaluate | simulate
    name: Mapped[str] = mapped_column(String(128), default='')
    # queued | provisioning | running | uploading | succeeded | failed | cancelled
    status: Mapped[str] = mapped_column(String(16), default='queued', index=True)
    spec: Mapped[dict] = mapped_column(JSON, default=dict)
    backend: Mapped[str] = mapped_column(String(16), default='local')
    pod_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    pod_info: Mapped[dict] = mapped_column(JSON, default=dict)
    token_hash: Mapped[str] = mapped_column(String(64), default='')
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    message: Mapped[str] = mapped_column(Text, default='')
    result: Mapped[dict] = mapped_column(JSON, default=dict)
    output_model_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    cost_per_hr: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_heartbeat: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    owner: Mapped[User] = relationship()


class JobLog(Base):
    __tablename__ = 'job_logs'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey('jobs.id', ondelete='CASCADE'), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    line: Mapped[str] = mapped_column(Text)


class JobMetric(Base):
    __tablename__ = 'job_metrics'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey('jobs.id', ondelete='CASCADE'), index=True)
    name: Mapped[str] = mapped_column(String(128), index=True)
    step: Mapped[int] = mapped_column(Integer, default=0)
    epoch: Mapped[int | None] = mapped_column(Integer, nullable=True)
    value: Mapped[float] = mapped_column(Float)


class JobArtifact(Base):
    __tablename__ = 'job_artifacts'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey('jobs.id', ondelete='CASCADE'), index=True)
    name: Mapped[str] = mapped_column(String(255))
    storage_key: Mapped[str] = mapped_column(String(512))
    content_type: Mapped[str] = mapped_column(String(128), default='application/octet-stream')
    size_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class SimSession(Base):
    """An interactive simulation: a GPU worker holding a model in memory."""

    __tablename__ = 'sim_sessions'

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    owner_id: Mapped[str] = mapped_column(ForeignKey('users.id', ondelete='CASCADE'), index=True)
    model_id: Mapped[str] = mapped_column(String(32))
    dataset_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # starting | ready | stopped | failed
    status: Mapped[str] = mapped_column(String(16), default='starting', index=True)
    backend: Mapped[str] = mapped_column(String(16), default='local')
    pod_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    endpoint_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    token_hash: Mapped[str] = mapped_column(String(64), default='')
    # Shared secret the platform presents to the worker's HTTP server.
    worker_secret: Mapped[str] = mapped_column(String(128), default='')
    spec: Mapped[dict] = mapped_column(JSON, default=dict)
    info: Mapped[dict] = mapped_column(JSON, default=dict)
    message: Mapped[str] = mapped_column(Text, default='')
    cost_per_hr: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    last_used_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    stopped_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
