import os
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'worker'))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


@pytest.fixture(scope='session')
def live_server(tmp_path_factory):
    """A real uvicorn server with the local launcher and mock workers."""
    port = _free_port()
    data = tmp_path_factory.mktemp('studio')
    os.environ.update({
        'HJS_DATA_DIR': str(data),
        'HJS_PUBLIC_URL': f'http://127.0.0.1:{port}',
        'HJS_LAUNCHER': 'local',
        'HJS_WORKER_MOCK': '1',
        'HJS_REAPER_INTERVAL_S': '1',
        'HJS_MAX_CONCURRENT_JOBS': '5',
        'HJW_WORKDIR': str(data / 'worker'),
        'HJW_MOCK_STEP_S': '0.01',
        'HJW_HEARTBEAT_S': '0.3',
    })
    from hjepa_studio.config import get_settings
    get_settings.cache_clear()
    import uvicorn
    from hjepa_studio.app import create_app

    server = uvicorn.Server(uvicorn.Config(create_app(), host='127.0.0.1', port=port,
                                           log_level='warning'))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield f'http://127.0.0.1:{port}'
    server.should_exit = True
    thread.join(timeout=5)
