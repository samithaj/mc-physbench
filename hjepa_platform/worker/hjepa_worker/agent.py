"""Worker entry point (``python -m hjepa_worker.agent``), started on a RunPod GPU pod or
locally. HJW_MODE=job runs one train/evaluate/simulate job and exits; HJW_MODE=serve
hosts an interactive simulation session. Set HJW_MOCK=1 to fake the GPU work."""

from __future__ import annotations

import csv
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

from .client import PlatformClient, PlatformError

PROGRESS_BAR_RE = re.compile(r'(\d+%\|)|(it/s\])|(s/it\])')
HEARTBEAT_S = float(os.environ.get('HJW_HEARTBEAT_S', '10'))


class Cancelled(Exception):
    pass


class JobRunner:
    def __init__(self, client: PlatformClient, spec: dict):
        self.client = client
        self.spec = spec
        self.params = spec['params']
        self.kind = spec['kind']
        self.progress = 0.0
        self.message = 'starting'
        self.cancelled = threading.Event()
        self.proc: subprocess.Popen | None = None
        self._lines: list[str] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads = [threading.Thread(target=self._pump, daemon=True)]
        self.workdir = Path(os.environ.get('HJW_WORKDIR', '/workspace/hjw')) / spec['job_id']
        self.workdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ plumbing

    def log(self, line: str) -> None:
        print(line, flush=True)
        with self._lock:
            self._lines.append(line)

    def status(self, message: str, progress: float | None = None) -> None:
        self.message = message
        if progress is not None:
            self.progress = progress
        self.log(f'[status] {message}')

    def _flush_logs(self) -> None:
        with self._lock:
            lines, self._lines = self._lines, []
        if lines:
            try:
                self.client.logs(lines)
            except PlatformError as exc:
                print(f'log upload failed: {exc}', file=sys.stderr)

    def _pump(self) -> None:
        last_beat = 0.0
        while not self._stop.is_set():
            self._flush_logs()
            if time.time() - last_beat > HEARTBEAT_S:
                last_beat = time.time()
                try:
                    reply = self.client.heartbeat(progress=round(self.progress, 4),
                                                  message=self.message[:2000])
                    if reply.get('cancel'):
                        self.cancel()
                except PlatformError as exc:
                    print(f'heartbeat failed: {exc}', file=sys.stderr)
            self._stop.wait(min(3.0, HEARTBEAT_S))
        self._flush_logs()

    def cancel(self) -> None:
        if not self.cancelled.is_set():
            self.cancelled.set()
            self.log('[status] cancellation requested')
            if self.proc and self.proc.poll() is None:
                self.proc.send_signal(signal.SIGINT)
                try:
                    self.proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.proc.kill()

    def check_cancel(self) -> None:
        if self.cancelled.is_set():
            raise Cancelled()

    def run_cmd(self, cmd: list[str], cwd: Path, poll=None, poll_every: float = 15.0) -> int:
        """Run a command, stream its output to the platform, call `poll` periodically."""
        self.log('$ ' + ' '.join(cmd))
        env = {**os.environ, 'PYTHONUNBUFFERED': '1', 'WANDB_MODE': 'disabled',
               'MUJOCO_GL': os.environ.get('MUJOCO_GL', 'egl')}
        self.proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, bufsize=0)
        reader = threading.Thread(target=self._read_output, args=(self.proc.stdout,), daemon=True)
        reader.start()
        while self.proc.poll() is None:
            time.sleep(poll_every if poll else 2)
            if poll:
                try:
                    poll()
                except Exception as exc:  # metrics tailing must never kill the run
                    print(f'poll failed: {exc}', file=sys.stderr)
        reader.join(timeout=10)
        if poll:
            poll()
        return self.proc.returncode

    def _read_output(self, stream) -> None:
        buf = b''
        last_bar = 0.0
        while chunk := stream.read(4096):
            buf += chunk
            parts = re.split(rb'[\r\n]', buf)
            buf = parts.pop()
            for raw in parts:
                line = raw.decode(errors='replace').rstrip()
                if not line:
                    continue
                if PROGRESS_BAR_RE.search(line):  # tqdm: keep one update every 15 s
                    if time.time() - last_bar < 15:
                        continue
                    last_bar = time.time()
                self.log(line)
        if buf.strip():
            self.log(buf.decode(errors='replace'))

    def upload(self, name: str, path: Path, content_type: str = 'application/octet-stream') -> str | None:
        if not path.exists():
            return None
        self.log(f'uploading {name} ({path.stat().st_size / 1e6:.1f} MB)')
        return self.client.upload_artifact(name, path, content_type)

    # ------------------------------------------------------------------ main

    def run(self) -> None:
        for t in self._threads:
            t.start()
        status, message, result = 'failed', '', {}
        try:
            if os.environ.get('HJW_MOCK') == '1':
                from . import mock
                result = mock.run_job(self)
            elif self.kind == 'train':
                result = self.train()
            elif self.kind == 'evaluate':
                result = self.evaluate()
            elif self.kind == 'simulate':
                result = self.simulate()
            else:
                raise ValueError(f'unknown job kind {self.kind!r}')
            status, message = 'succeeded', result.pop('message', 'done')
        except Cancelled:
            status, message = 'cancelled', 'cancelled by user'
        except Exception as exc:
            self.log(traceback.format_exc())
            message = f'{type(exc).__name__}: {exc}'
        finally:
            self._stop.set()
            for t in self._threads:
                t.join(timeout=15)
        self.client.complete(status, message, result)

    # ------------------------------------------------------------------ train

    def train(self) -> dict:
        from .inputs import ensure_val_split, fetch_dataset
        from .recipes import (H_JEPA_DIR, fetch_eval_tasks, h5_shape_info, run_dir_for,
                              train_command, write_custom_configs)

        p = self.params
        datasets = {d['role']: d for d in self.spec['inputs']['datasets']}
        self.status('preparing training data', 0.02)
        train = fetch_dataset(datasets['train'], log=self.log)
        self.check_cancel()
        official = datasets['train']['source'] == 'official'
        if official:
            train_name, val_name = train['train'], train['val']
        elif 'val' in datasets:
            train_name, val_name = train['train'], fetch_dataset(datasets['val'], log=self.log)['train']
        else:
            train_name, val_name = ensure_val_split(train, log=self.log)

        levels = int(p['levels'])
        if p['preset'] == 'custom':
            info = h5_shape_info(train['path'])
            self.log(f'custom data: {info["img_size"]}px frames, action dim {info["action_dim"]}')
            config_name = write_custom_configs(p['variant'], train_name, val_name, info['img_size'],
                                               info['action_dim'], int(p['frameskip']), levels)
        else:
            config_name = f'{p["preset"]}_{p["variant"]}'
        if p.get('run_planning_eval'):
            fetch_eval_tasks(log=self.log)
        self.check_cancel()

        params = {**p, 'gpu_count': self.spec.get('gpu', {}).get('count', 1)}
        cmd = train_command(params, config_name, train_name, val_name, official, levels)
        run_dir = run_dir_for(p, config_name)
        tail = MetricsTail(run_dir, self, epochs=int(p['epochs']))
        self.status(f'training {config_name} for {p["epochs"]} epochs', 0.05)
        code = self.run_cmd(cmd, cwd=H_JEPA_DIR, poll=tail.poll)
        self.check_cancel()
        if code != 0:
            raise RuntimeError(f'main_hjepa.py exited with code {code}')
        self.status('uploading checkpoint and results', 0.95)
        return self._collect_train_outputs(run_dir, config_name, tail)

    def _collect_train_outputs(self, run_dir: Path, config_name: str, tail: 'MetricsTail') -> dict:
        import yaml

        files = {}
        ckpt = run_dir / f'{config_name}_object.ckpt'
        if not ckpt.exists():
            found = sorted(run_dir.glob('*_object.ckpt'))
            ckpt = found[-1] if found else ckpt
        if self.upload('model_object.ckpt', ckpt):
            files['object_ckpt'] = 'model_object.ckpt'
        for name, logical in (('normalizer.pt', 'normalizer'), ('config.yaml', 'config')):
            if self.upload(name, run_dir / name):
                files[logical] = name
        probe_dir = run_dir / 'final_probing_decoding_eval'
        for dec in sorted(probe_dir.glob('decoder_level*.ckpt')):
            if self.upload(dec.name, dec):
                files[dec.stem] = dec.name
        for png in sorted(probe_dir.rglob('*.png'))[:12]:
            self.upload(f'decoding_{png.stem}.png', png, 'image/png')
        metrics: dict = {'final': tail.last_values()}
        probe_metrics = probe_dir / 'metrics.yaml'
        if probe_metrics.exists():
            self.upload('probing_metrics.yaml', probe_metrics, 'text/yaml')
            metrics['probing'] = _floats(yaml.safe_load(probe_metrics.read_text()) or {})
        planning = sorted((run_dir / 'planning_eval').rglob('metrics.yaml'))
        if planning:
            self.upload('planning_metrics.yaml', planning[-1], 'text/yaml')
            pm = yaml.safe_load(planning[-1].read_text()) or {}
            metrics['planning'] = {self.params['planner']: _floats(pm)}
        if tail.csv_path:
            self.upload('metrics.csv', tail.csv_path, 'text/csv')
        if 'object_ckpt' not in files:
            raise RuntimeError(f'training produced no checkpoint in {run_dir}')
        msg = 'training finished'
        sr = (metrics.get('planning') or {}).get(self.params['planner'], {}).get('success_rate')
        if sr is not None:
            msg += f'; planning success {sr:.1f}%'
        return {'model_files': files, 'metrics': metrics, 'message': msg}

    # ------------------------------------------------------------------ evaluate

    def evaluate(self) -> dict:
        import yaml

        from .inputs import fetch_model
        from .recipes import H_JEPA_DIR, eval_command, fetch_eval_tasks

        p = self.params
        self.status('downloading model', 0.05)
        model = fetch_model(self.spec['inputs']['model'], log=self.log)
        fetch_eval_tasks(log=self.log)
        self.check_cancel()
        self.status(f'planning eval: {p["env"]}_{p["planner"]}, {p["num_eval"]} tasks', 0.15)
        code = self.run_cmd(eval_command(p['env'], p['planner'], model['ckpt'], p['seed'],
                                         p['num_eval']), cwd=H_JEPA_DIR)
        self.check_cancel()
        if code != 0:
            raise RuntimeError(f'eval.py exited with code {code}')
        out = model['ckpt'].parent / 'platform_eval' / 'metrics.yaml'
        metrics = _floats(yaml.safe_load(out.read_text()) or {}) if out.exists() else {}
        self.upload('metrics.yaml', out, 'text/yaml')
        sr = metrics.get('success_rate')
        msg = f'success rate {sr:.1f}%' if sr is not None else 'evaluation finished'
        return {'metrics': metrics, 'message': msg}

    # ------------------------------------------------------------------ simulate

    def simulate(self) -> dict:
        from .inputs import fetch_dataset, fetch_model
        from .render import comparison_gif, comparison_grid
        from .simulator import WorldModelSimulator, load_episode

        p = self.params
        self.status('downloading model and data', 0.05)
        model = fetch_model(self.spec['inputs']['model'], log=self.log)
        ds = fetch_dataset(self.spec['inputs']['datasets'][0], log=self.log)
        h5 = ds.get('val_path') or ds['path']
        self.check_cancel()
        self.status('loading world model', 0.2)
        sim = WorldModelSimulator(model['ckpt'], model['normalizer'], model['config'], model['decoders'])
        summary = {'episodes': [], 'decoders': sorted(sim.decoders), 'frameskip': sim.frameskip}
        for k, ep_idx in enumerate(p['episodes']):
            self.check_cancel()
            self.status(f'dreaming episode {ep_idx}', 0.25 + 0.7 * k / len(p['episodes']))
            ep = load_episode(h5, ep_idx)
            out = sim.dream(ep, p['levels'], p['start'], p['context'], p['horizon'],
                            p['action_mode'], p['action_scale'], p['seed'])
            rows = []
            ep_summary = {'episode': ep_idx, 'levels': {}}
            for lv, r in out.items():
                gif = comparison_gif(r['context_frames'], r['real_frames'], r['pred_frames'], f'L{lv}')
                path = self.workdir / f'ep{ep_idx}_level{lv}.gif'
                path.write_bytes(gif)
                self.upload(path.name, path, 'image/gif')
                rows += [(f'L{lv} real', list(r['real_frames'])), (f'L{lv} dream', list(r['pred_frames']))]
                ep_summary['levels'][lv] = {k2: r[k2] for k2 in
                                            ('latent_error', 'pixel_error', 'raw_steps', 'decoded_with')}
            grid = self.workdir / f'ep{ep_idx}_grid.png'
            grid.write_bytes(comparison_grid(rows))
            self.upload(grid.name, grid, 'image/png')
            summary['episodes'].append(ep_summary)
        out_json = self.workdir / 'simulation.json'
        out_json.write_text(json.dumps(summary))
        self.upload('simulation.json', out_json, 'application/json')
        return {'metrics': {'simulation': _sim_headline(summary)}, 'summary': _sim_headline(summary),
                'message': f'dreamed {len(p["episodes"])} episode(s)'}


def _floats(d: dict) -> dict:
    return {k: float(v) for k, v in d.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


def _sim_headline(summary: dict) -> dict:
    """Mean normalized latent error per level at the first and last dreamed step."""
    out = {}
    for ep in summary['episodes']:
        for lv, r in ep['levels'].items():
            errs = r['latent_error']
            if errs:
                agg = out.setdefault(f'level{lv}', {'first': [], 'last': []})
                agg['first'].append(errs[0])
                agg['last'].append(errs[-1])
    return {lv: {k: sum(v) / len(v) for k, v in agg.items()} for lv, agg in out.items()}


class MetricsTail:
    """Follow Lightning's CSVLogger output and forward new rows as metrics."""

    def __init__(self, run_dir: Path, runner: JobRunner, epochs: int):
        self.run_dir = run_dir
        self.runner = runner
        self.epochs = max(1, epochs)
        self.sent_rows = 0
        self.csv_path: Path | None = None
        self.latest: dict[str, float] = {}

    def _find(self) -> Path | None:
        found = sorted((self.run_dir / 'lightning_logs').glob('version_*/metrics.csv'),
                       key=lambda p: p.stat().st_mtime)
        return found[-1] if found else None

    def poll(self) -> None:
        path = self._find()
        if path is None:
            return
        if path != self.csv_path:
            self.csv_path, self.sent_rows = path, 0
        with open(path, newline='') as fh:
            rows = list(csv.DictReader(fh))
        new = rows[self.sent_rows:]
        self.sent_rows = len(rows)
        out, max_epoch = [], None
        for row in new:
            step = int(float(row.get('step') or 0))
            epoch = int(float(row['epoch'])) if row.get('epoch') else None
            if epoch is not None:
                max_epoch = epoch if max_epoch is None else max(max_epoch, epoch)
            for name, value in row.items():
                if name in ('step', 'epoch') or value in (None, ''):
                    continue
                try:
                    v = float(value)
                except ValueError:
                    continue
                self.latest[name] = v
                out.append({'name': name, 'value': v, 'step': step, 'epoch': epoch})
        if out:
            self.runner.client.metrics(out)
        if max_epoch is not None:
            frac = min(1.0, (max_epoch + 1) / self.epochs)
            self.runner.progress = 0.05 + 0.85 * frac
            self.runner.message = f'training: epoch {max_epoch + 1}/{self.epochs}'

    def last_values(self) -> dict:
        return dict(self.latest)


def main() -> None:
    mode = os.environ.get('HJW_MODE', 'job')
    client = PlatformClient.from_env()
    if mode == 'serve':
        from .server import serve
        serve(client)
        return
    try:
        spec = client.spec()
    except PlatformError as exc:
        print(f'could not fetch job spec: {exc}', file=sys.stderr)
        sys.exit(1)
    JobRunner(client, spec).run()


if __name__ == '__main__':
    main()
