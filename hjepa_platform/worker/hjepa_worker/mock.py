"""Fake GPU work (HJW_MOCK=1): exercises the whole platform loop without torch or H-JEPA.

Frames are a synthetic ball moving under the actions; the "world model" dreams a copy
that drifts, so latent errors grow over the horizon like a real rollout.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import random
import time

import numpy as np

STEP_S = float(os.environ.get('HJW_MOCK_STEP_S', '0.05'))
SIZE = 64


def _ball_frame(x: float, y: float, hue: int = 0) -> np.ndarray:
    img = np.zeros((SIZE, SIZE, 3), np.uint8)
    img[..., 2] = 40
    yy, xx = np.mgrid[:SIZE, :SIZE]
    mask = (xx - x) ** 2 + (yy - y) ** 2 < 36
    img[mask] = (250, 180 - hue, 60 + hue)
    return img


def _trajectory(n: int, seed: int, drift: float = 0.0):
    rng = random.Random(seed)
    x, y, vx, vy = 20.0, 30.0, 1.6, 1.1
    out = []
    for t in range(n):
        x, y = x + vx + drift * t * 0.05, y + vy - drift * t * 0.03
        if not 6 < x < SIZE - 6:
            vx = -vx
        if not 6 < y < SIZE - 6:
            vy = -vy
        x, y = min(max(x, 6), SIZE - 6), min(max(y, 6), SIZE - 6)
        out.append((x + rng.uniform(-0.3, 0.3), y + rng.uniform(-0.3, 0.3)))
    return out


def run_job(runner) -> dict:
    kind, p = runner.kind, runner.params
    runner.status(f'[mock] preparing {kind}', 0.02)
    time.sleep(STEP_S)
    if kind == 'train':
        return _train(runner, p)
    if kind == 'evaluate':
        runner.check_cancel()
        sr = round(random.Random(p.get('seed', 0)).uniform(20, 80), 1)
        path = runner.workdir / 'metrics.yaml'
        path.write_text(f'success_rate: {sr}\nnum_eval: {p["num_eval"]}\n')
        runner.upload('metrics.yaml', path, 'text/yaml')
        return {'metrics': {'success_rate': sr, 'num_eval': float(p['num_eval'])},
                'message': f'[mock] success rate {sr:.1f}%'}
    if kind == 'simulate':
        return _simulate(runner, p)
    raise ValueError(kind)


def _train(runner, p: dict) -> dict:
    levels, epochs = int(p['levels']), int(p['epochs'])
    steps_per_epoch = 5
    for epoch in range(epochs):
        rows = []
        for s in range(steps_per_epoch):
            runner.check_cancel()
            step = epoch * steps_per_epoch + s
            for lv in range(1, levels + 1):
                loss = 1.5 * math.exp(-0.15 * step / lv) + 0.05 * lv + random.uniform(0, 0.03)
                rows.append({'name': f'fit/pred_loss_level{lv}', 'value': loss, 'step': step, 'epoch': epoch})
                rows.append({'name': f'fit/sigreg_loss_level{lv}', 'value': 0.3 * loss, 'step': step, 'epoch': epoch})
            time.sleep(STEP_S)
        rows.append({'name': 'validate/pred_loss_level1', 'value': rows[-2]['value'] * 1.1,
                     'step': (epoch + 1) * steps_per_epoch, 'epoch': epoch})
        runner.client.metrics(rows)
        runner.progress = 0.05 + 0.85 * (epoch + 1) / epochs
        runner.status(f'[mock] epoch {epoch + 1}/{epochs}')
    files = {}
    for name, logical in (('model_object.ckpt', 'object_ckpt'), ('normalizer.pt', 'normalizer'),
                          ('config.yaml', 'config')):
        path = runner.workdir / name
        path.write_bytes(f'mock {logical} for {p["preset"]}_{p["variant"]}\n'.encode())
        runner.upload(name, path)
        files[logical] = name
    if p.get('train_decoders', True):
        for lv in range(1, levels + 1):
            path = runner.workdir / f'decoder_level{lv}.ckpt'
            path.write_bytes(b'mock decoder')
            runner.upload(path.name, path)
            files[f'decoder_level{lv}'] = path.name
    metrics = {'final': {f'fit/pred_loss_level{lv}': 0.1 * lv for lv in range(1, levels + 1)}}
    if p.get('run_planning_eval'):
        metrics['planning'] = {p['planner']: {'success_rate': round(30 + 10 * levels, 1)}}
    return {'model_files': files, 'metrics': metrics, 'message': '[mock] training finished'}


def _simulate(runner, p: dict) -> dict:
    from .render import comparison_gif, comparison_grid

    summary = {'episodes': [], 'decoders': list(p['levels']), 'frameskip': 5, 'mock': True}
    for ep_idx in p['episodes']:
        runner.check_cancel()
        n = p['context'] + p['horizon']
        real = [_ball_frame(x, y) for x, y in _trajectory(n, ep_idx)]
        ep_summary = {'episode': ep_idx, 'levels': {}}
        rows = []
        for lv in p['levels']:
            dream = [_ball_frame(x, y, hue=40 * lv)
                     for x, y in _trajectory(n, ep_idx, drift=1.0 / lv)][p['context']:]
            ctx = real[:p['context']]
            gif = comparison_gif(np.array(ctx), np.array(real[p['context']:]), np.array(dream), f'L{lv}')
            path = runner.workdir / f'ep{ep_idx}_level{lv}.gif'
            path.write_bytes(gif)
            runner.upload(path.name, path, 'image/gif')
            errs = [0.02 * (t + 1) / lv for t in range(len(dream))]
            ep_summary['levels'][lv] = {'latent_error': errs, 'pixel_error': errs,
                                        'raw_steps': list(range(len(dream))),
                                        'decoded_with': 'mock'}
            rows += [(f'L{lv} real', real[p['context']:]), (f'L{lv} dream', dream)]
        grid = runner.workdir / f'ep{ep_idx}_grid.png'
        grid.write_bytes(comparison_grid(rows))
        runner.upload(grid.name, grid, 'image/png')
        summary['episodes'].append(ep_summary)
    out = runner.workdir / 'simulation.json'
    out.write_text(json.dumps(summary))
    runner.upload('simulation.json', out, 'application/json')
    return {'metrics': {}, 'summary': {}, 'message': f'[mock] dreamed {len(p["episodes"])} episode(s)'}


def _b64(frame: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(frame).resize((192, 192), Image.NEAREST).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


class MockEngine:
    def __init__(self, spec: dict):
        self.levels = int(spec['params'].get('num_levels', 2))
        self.pos = {}
        self.ball = {}
        self.real = []
        self.episode = 0

    def info(self, _=None) -> dict:
        return {'num_levels': self.levels, 'frameskip': 5, 'episodes': 10,
                'decoders': list(range(1, self.levels + 1)), 'history': [3] * self.levels,
                'action_dim': 2, 'action_low': [-1, -1], 'action_high': [1, 1], 'mock': True,
                'episode': self.episode}

    def reset(self, body: dict) -> dict:
        self.episode = int(body.get('episode', 0))
        start = int(body.get('start', 0))
        self.real = _trajectory(400, self.episode)
        self.pos = {lv: start for lv in range(1, self.levels + 1)}
        self.ball = {lv: list(self.real[start]) for lv in range(1, self.levels + 1)}
        frame = _b64(_ball_frame(*self.real[start]))
        return {'episode': self.episode, 'episode_len': 400, 'real': frame,
                'levels': {str(lv): frame for lv in self.ball}, **self.info()}

    def step(self, body: dict) -> dict:
        lv = int(body.get('level', 1))
        if lv not in self.ball:
            raise ValueError('call reset first' if not self.ball else f'level must be 1..{self.levels}')
        stride = 2 ** (lv - 1)
        self.pos[lv] += stride
        mode = body.get('mode', 'ground_truth')
        target = self.real[min(self.pos[lv], len(self.real) - 1)]
        if mode == 'user' and body.get('action'):
            ax, ay = body['action'][:2]
            self.ball[lv] = [self.ball[lv][0] + 3 * ax * stride, self.ball[lv][1] + 3 * ay * stride]
        else:
            self.ball[lv] = [0.8 * self.ball[lv][0] + 0.2 * target[0] + 0.3,
                             0.8 * self.ball[lv][1] + 0.2 * target[1]]
        self.ball[lv] = [min(max(v, 6), SIZE - 6) for v in self.ball[lv]]
        err = ((self.ball[lv][0] - target[0]) ** 2 + (self.ball[lv][1] - target[1]) ** 2) / 400
        return {'level': lv, 'dreamed': self.pos[lv], 'frame': _b64(_ball_frame(*self.ball[lv], hue=40 * lv)),
                'real': _b64(_ball_frame(*target)), 'latent_error': err, 'raw_step': self.pos[lv]}

    def rollout(self, body: dict) -> dict:
        return {'steps': [self.step(body) for _ in range(min(int(body.get('horizon', 8)), 64))]}

    def sample_action(self, body: dict) -> dict:
        return {'action': [round(random.uniform(-1, 1), 3), round(random.uniform(-1, 1), 3)]}
