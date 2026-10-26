"""End-to-end: accounts, dataset sharing, training/eval/simulation jobs and interactive
sessions, with the local launcher running mock workers as real subprocesses."""

import time

import h5py
import httpx
import numpy as np
import pytest


def _register(base, name):
    r = httpx.post(f'{base}/api/auth/register',
                   json={'email': f'{name}@example.com', 'username': name, 'password': 'hunter2hunter2'})
    assert r.status_code == 200, r.text
    return {'Authorization': f'Bearer {r.json()["access_token"]}'}


@pytest.fixture(scope='module')
def alice(live_server):
    return _register(live_server, 'alice')


@pytest.fixture(scope='module')
def bob(live_server):
    return _register(live_server, 'bob')


def _wait_job(base, h, job_id, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = httpx.get(f'{base}/api/jobs/{job_id}', headers=h).json()
        if job['status'] in ('succeeded', 'failed', 'cancelled'):
            return job
        time.sleep(0.3)
    raise AssertionError(f'job {job_id} did not finish: {job}')


def _make_h5(path, episodes=6, length=40, size=64, action_dim=3):
    n = episodes * length
    with h5py.File(path, 'w') as f:
        f['ep_len'] = np.full(episodes, length, np.int64)
        f['ep_offset'] = np.arange(episodes, dtype=np.int64) * length
        f['pixels'] = np.random.randint(0, 255, (n, size, size, 3), np.uint8)
        f['action'] = np.random.randn(n, action_dim).astype(np.float32)


def _upload(base, h, path, **meta):
    data = path.read_bytes()
    ds = httpx.post(f'{base}/api/datasets', headers=h,
                    json={'name': path.stem, 'size_bytes': len(data), **meta}).json()
    chunk = 100_000
    for off in range(0, len(data), chunk):
        r = httpx.put(f'{base}/api/datasets/{ds["id"]}/upload?offset={off}', headers=h,
                      content=data[off:off + chunk])
        assert r.status_code == 200, r.text
    r = httpx.post(f'{base}/api/datasets/{ds["id"]}/complete', headers=h)
    assert r.status_code == 200, r.text
    return r.json()


def test_auth(live_server, alice):
    r = httpx.post(f'{live_server}/api/auth/login', json={'login': 'alice', 'password': 'wrong-pass'})
    assert r.status_code == 401
    r = httpx.post(f'{live_server}/api/auth/login', json={'login': 'alice@example.com',
                                                           'password': 'hunter2hunter2'})
    assert r.status_code == 200
    me = httpx.get(f'{live_server}/api/auth/me', headers=alice).json()
    assert me['username'] == 'alice' and me['is_admin']  # first account administers
    assert httpx.get(f'{live_server}/api/auth/me').status_code == 401
    r = httpx.put(f'{live_server}/api/auth/settings', headers=alice,
                  json={'runpod_api_key': 'rpa_ABCDEFGHIJKLMNOPQRSTUVWX'})
    assert r.json()['has_runpod_key']


def test_catalog_seeded(live_server, alice):
    datasets = httpx.get(f'{live_server}/api/datasets?scope=public', headers=alice).json()
    assert {d['env_preset'] for d in datasets if d['source'] == 'official'} == {
        'pusht', 'ant', 'fourroom', 'cube'}
    models = httpx.get(f'{live_server}/api/models?env=ant', headers=alice).json()
    assert {m['variant'] for m in models} >= {'lewm', 'hjepa_l2', 'hjepa_l3', 'hjepa_l4'}


def test_dataset_upload_validation_and_sharing(live_server, alice, bob, tmp_path):
    good = tmp_path / 'balls.h5'
    _make_h5(good)
    ds = _upload(live_server, alice, good, env_preset='custom')
    assert ds['status'] == 'ready', ds['inspection']
    assert ds['inspection']['img_size'] == 64 and ds['inspection']['action_dim'] == 3
    assert ds['preview_url']
    assert httpx.get(live_server + ds['preview_url']).headers['content-type'] == 'image/png'

    # private by default: bob cannot see it
    assert httpx.get(f'{live_server}/api/datasets/{ds["id"]}', headers=bob).status_code == 404
    httpx.patch(f'{live_server}/api/datasets/{ds["id"]}', headers=alice, json={'visibility': 'public'})
    assert httpx.get(f'{live_server}/api/datasets/{ds["id"]}', headers=bob).status_code == 200
    # but bob cannot modify it
    r = httpx.patch(f'{live_server}/api/datasets/{ds["id"]}', headers=bob, json={'name': 'mine'})
    assert r.status_code == 404

    bad = tmp_path / 'bad.h5'
    with h5py.File(bad, 'w') as f:
        f['pixels'] = np.zeros((4, 8, 8, 3), np.uint8)
    ds_bad = _upload(live_server, alice, bad)
    assert ds_bad['status'] == 'invalid'
    assert any('ep_len' in e for e in ds_bad['inspection']['errors'])


def test_upload_offset_mismatch(live_server, alice):
    ds = httpx.post(f'{live_server}/api/datasets', headers=alice,
                    json={'name': 'x', 'size_bytes': 10}).json()
    r = httpx.put(f'{live_server}/api/datasets/{ds["id"]}/upload?offset=5', headers=alice, content=b'abc')
    assert r.status_code == 409
    r = httpx.put(f'{live_server}/api/datasets/{ds["id"]}/upload?offset=0', headers=alice, content=b'x' * 11)
    assert r.status_code == 413


def test_signed_urls(live_server, alice):
    r = httpx.get(f'{live_server}/api/files/datasets/x/data.h5?exp=9999999999&sig=deadbeef')
    assert r.status_code == 403
    r = httpx.put(f'{live_server}/api/files/datasets/x/data.h5?exp=9999999999&sig=deadbeef', content=b'1')
    assert r.status_code == 403


def test_train_evaluate_simulate_on_official_data(live_server, alice):
    ds = next(d for d in httpx.get(f'{live_server}/api/datasets?scope=public', headers=alice).json()
              if d['source'] == 'official' and d['env_preset'] == 'pusht')
    body = {'preset': 'pusht', 'variant': 'hjepa_l2', 'train_dataset_id': ds['id'], 'epochs': 3,
            'gpu': {'type': 'NVIDIA GeForce RTX 5090'}}
    est = httpx.post(f'{live_server}/api/jobs/estimate', headers=alice, json=body).json()
    assert est['usd'] > 0
    job = httpx.post(f'{live_server}/api/jobs/train', headers=alice, json=body).json()
    assert job['status'] == 'provisioning', job
    job = _wait_job(live_server, alice, job['id'])
    assert job['status'] == 'succeeded', job
    metrics = httpx.get(f'{live_server}/api/jobs/{job["id"]}/metrics', headers=alice).json()
    assert 'fit/pred_loss_level2' in metrics and len(metrics['fit/pred_loss_level1']) == 15
    logs = httpx.get(f'{live_server}/api/jobs/{job["id"]}/logs', headers=alice).json()
    assert any('epoch 3/3' in line['line'] for line in logs['lines'])

    model = httpx.get(f'{live_server}/api/models/{job["output_model_id"]}', headers=alice).json()
    assert model['num_levels'] == 2 and 'object_ckpt' in model['files']
    assert model['metrics']['planning']['l2']['success_rate'] == 50.0
    url = httpx.get(f'{live_server}/api/models/{model["id"]}/files/object_ckpt', headers=alice).json()['url']
    assert httpx.get(live_server + url).content.startswith(b'mock object_ckpt')

    ev = httpx.post(f'{live_server}/api/jobs/evaluate', headers=alice,
                    json={'model_id': model['id'], 'planner': 'l2_project'}).json()
    ev = _wait_job(live_server, alice, ev['id'])
    assert ev['status'] == 'succeeded', ev
    model = httpx.get(f'{live_server}/api/models/{model["id"]}', headers=alice).json()
    assert 'l2_project' in model['metrics']['planning']
    board = httpx.get(f'{live_server}/api/models/leaderboard?env=pusht', headers=alice).json()
    assert board and board[0]['model']['id'] == model['id']

    sim = httpx.post(f'{live_server}/api/jobs/simulate', headers=alice,
                     json={'model_id': model['id'], 'episodes': [0, 1], 'horizon': 6}).json()
    sim = _wait_job(live_server, alice, sim['id'])
    assert sim['status'] == 'succeeded', sim
    arts = {a['name']: a for a in httpx.get(f'{live_server}/api/jobs/{sim["id"]}/artifacts',
                                             headers=alice).json()}
    assert {'ep0_level1.gif', 'ep0_level2.gif', 'ep1_grid.png', 'simulation.json'} <= set(arts)
    assert httpx.get(live_server + arts['ep0_level1.gif']['url']).content[:3] == b'GIF'


def test_bad_requests(live_server, alice, bob):
    ds = next(d for d in httpx.get(f'{live_server}/api/datasets?scope=public', headers=alice).json()
              if d['env_preset'] == 'ant')
    r = httpx.post(f'{live_server}/api/jobs/train', headers=alice,
                   json={'preset': 'pusht', 'variant': 'lewm', 'train_dataset_id': ds['id']})
    assert r.status_code == 422  # ant data for a pusht run
    r = httpx.post(f'{live_server}/api/jobs/train', headers=alice,
                   json={'preset': 'ant', 'variant': 'lewm', 'train_dataset_id': ds['id'],
                         'extra_overrides': ['hydra.run.dir=/tmp']})
    assert r.status_code == 422
    r = httpx.post(f'{live_server}/api/jobs/train', headers=alice,
                   json={'preset': 'ant', 'variant': 'lewm', 'train_dataset_id': ds['id'],
                         'gpu': {'type': 'NVIDIA H100'}})
    assert r.status_code == 422
    official = next(m for m in httpx.get(f'{live_server}/api/models?env=ant&variant=lewm',
                                         headers=alice).json())
    r = httpx.post(f'{live_server}/api/jobs/evaluate', headers=alice,
                   json={'model_id': official['id'], 'planner': 'l3'})
    assert r.status_code == 422  # a flat model has no level-3 planner
    jobs = httpx.get(f'{live_server}/api/jobs', headers=alice).json()
    assert httpx.get(f'{live_server}/api/jobs/{jobs[0]["id"]}', headers=bob).status_code == 404


def test_custom_data_training_and_cancel(live_server, alice, tmp_path):
    path = tmp_path / 'custom.h5'
    _make_h5(path)
    ds = _upload(live_server, alice, path)
    job = httpx.post(f'{live_server}/api/jobs/train', headers=alice,
                     json={'preset': 'custom', 'variant': 'hjepa_l3', 'train_dataset_id': ds['id'],
                           'epochs': 200, 'run_planning_eval': True}).json()
    assert job['spec']['run_planning_eval'] is False  # no simulator for custom data
    for _ in range(100):
        if httpx.get(f'{live_server}/api/jobs/{job["id"]}', headers=alice).json()['last_heartbeat']:
            break
        time.sleep(0.1)
    httpx.post(f'{live_server}/api/jobs/{job["id"]}/cancel', headers=alice)
    job = _wait_job(live_server, alice, job['id'])
    assert job['status'] == 'cancelled', job
    assert job['output_model_id'] is None


def test_worker_token_scoping(live_server, alice):
    jobs = httpx.get(f'{live_server}/api/jobs', headers=alice).json()
    r = httpx.get(f'{live_server}/api/worker/jobs/{jobs[0]["id"]}/spec',
                  headers={'Authorization': 'Bearer not-the-token'})
    assert r.status_code == 403


def test_interactive_session(live_server, alice):
    model = next(m for m in httpx.get(f'{live_server}/api/models?env=pusht&variant=hjepa_l2',
                                      headers=alice).json())
    sess = httpx.post(f'{live_server}/api/sessions', headers=alice, json={'model_id': model['id']}).json()
    for _ in range(100):
        sess = httpx.get(f'{live_server}/api/sessions/{sess["id"]}', headers=alice).json()
        if sess['status'] != 'starting':
            break
        time.sleep(0.1)
    assert sess['status'] == 'ready', sess
    call = f'{live_server}/api/sessions/{sess["id"]}/call'
    r = httpx.post(f'{call}/reset', headers=alice, json={'episode': 2, 'start': 5}).json()
    assert r['episode'] == 2 and r['real']
    step = httpx.post(f'{call}/step', headers=alice,
                      json={'level': 1, 'mode': 'user', 'action': [0.5, -0.5]}).json()
    assert step['frame'] and step['latent_error'] is not None
    roll = httpx.post(f'{call}/rollout', headers=alice, json={'level': 2, 'horizon': 4}).json()
    assert len(roll['steps']) == 4
    assert httpx.post(f'{call}/shutdown', headers=alice).status_code == 404
    httpx.post(f'{live_server}/api/sessions/{sess["id"]}/stop', headers=alice)
    r = httpx.post(f'{call}/step', headers=alice, json={'level': 1})
    assert r.status_code == 409


def test_worker_crash_loop_guard(live_server, alice):
    from hjepa_studio.db import session_factory
    from hjepa_studio.models import Job, User
    from hjepa_studio.security import new_worker_token

    db = session_factory()()
    user = db.query(User).filter_by(username='alice').one()
    token, digest = new_worker_token()
    job = Job(owner_id=user.id, kind='train', status='provisioning', token_hash=digest,
              spec={'gpu': {}, 'max_hours': 1})
    db.add(job)
    db.commit()
    hdr = {'Authorization': f'Bearer {token}'}
    for _ in range(3):  # container restarts re-fetch the spec
        assert httpx.get(f'{live_server}/api/worker/jobs/{job.id}/spec', headers=hdr).status_code == 200
    assert httpx.get(f'{live_server}/api/worker/jobs/{job.id}/spec', headers=hdr).status_code == 409
    job = httpx.get(f'{live_server}/api/jobs/{job.id}', headers=alice).json()
    assert job['status'] == 'failed' and 'crashed' in job['message']
    db.close()
