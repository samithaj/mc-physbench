"""H-JEPA Studio: share world-model data, train hierarchical JEPA models on rented GPUs,
and simulate them in the browser."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .db import init_engine, session_factory
from .reaper import reaper_loop
from .routers import auth, datasets, files, jobs, meta, sessions, world_models, worker
from .services import seed_catalog

STATIC_DIR = Path(__file__).parent / 'static'


def create_app(run_reaper: bool = True) -> FastAPI:
    settings = get_settings()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI):
        init_engine()
        if settings.seed_catalog:
            db = session_factory()()
            try:
                seed_catalog(db)
            finally:
                db.close()
        task = asyncio.create_task(reaper_loop()) if run_reaper else None
        yield
        if task:
            task.cancel()

    app = FastAPI(title='H-JEPA Studio', version='0.1.0', lifespan=lifespan)
    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins,
                           allow_methods=['*'], allow_headers=['*'])
    for module in (auth, datasets, world_models, jobs, sessions, worker, files, meta):
        app.include_router(module.router)

    @app.get('/healthz')
    def healthz():
        return {'ok': True}

    app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')

    @app.get('/', include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / 'index.html')

    return app


app = create_app()
