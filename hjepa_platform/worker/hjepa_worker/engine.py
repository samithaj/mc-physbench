"""Stateful interactive simulation on top of WorldModelSimulator.

The browser picks an episode and start frame (reset), then advances any level one
predicted step at a time (step) with an action of its choosing:
  level 1: the recorded action, a user-set raw action (held for `frameskip` env steps),
           zero, or random;
  level N: the recorded macro-action, or one sampled from the level's latent-action queue
           (the macro-actions it saw in training).
"""

from __future__ import annotations

import base64

import numpy as np

from .simulator import WorldModelSimulator, episode_count, load_episode, png_bytes

MAX_ROLLOUT = 64


def _b64png(frame: np.ndarray) -> str:
    return base64.b64encode(png_bytes(frame, size=192)).decode()


class InteractiveEngine:
    def __init__(self, sim: WorldModelSimulator, h5_path, meta: dict):
        self.sim = sim
        self.h5 = h5_path
        self.meta = meta
        self.n_episodes = episode_count(h5_path)
        self.ep = None
        self.gt = None
        self.state: dict[int, dict] = {}

    def info(self, _: dict | None = None) -> dict:
        action_dim = None
        lo = hi = None
        if self.ep is not None:
            action_dim = int(self.ep.actions.shape[1])
            lo, hi = self.ep.actions.min(0).tolist(), self.ep.actions.max(0).tolist()
        return {**self.meta, 'num_levels': self.sim.num_levels, 'frameskip': self.sim.frameskip,
                'episodes': self.n_episodes, 'decoders': sorted(self.sim.decoders),
                'history': self.sim.history, 'action_dim': action_dim,
                'action_low': lo, 'action_high': hi,
                'episode': None if self.ep is None else self.meta.get('episode')}

    def reset(self, body: dict) -> dict:
        torch = self.sim.torch
        episode = int(body.get('episode', 0))
        start_frame = int(body.get('start', 0))
        self.ep = load_episode(self.h5, episode)
        self.meta['episode'] = episode
        self.gt = self.sim.encode(self.ep)
        self.state = {}
        for lv in range(1, self.sim.num_levels + 1):
            fidx = self.gt['frame_index'][lv]
            pos = int(min(np.searchsorted(fidx, start_frame), len(fidx) - 1))
            ctx = self.sim.history[lv - 1]
            first = max(0, pos - ctx + 1)
            self.state[lv] = {
                'latents': self.gt['embeds'][lv][:, first:pos + 1].clone(),
                'acts': [self.gt['act_embs'][lv][:, i:i + 1] for i in range(first, pos)],
                'gt_pos': pos,  # GT index aligned with the newest latent
                'dreamed': 0,
            }
        with torch.no_grad():
            frames = {str(lv): _b64png(self._frame(lv, s['latents'][0, -1:])[0])
                      for lv, s in self.state.items()}
        return {'episode': episode, 'episode_len': int(len(self.ep.pixels)),
                'real': _b64png(self.ep.pixels[min(start_frame, len(self.ep.pixels) - 1)]),
                'levels': frames, **self.info()}

    def _frame(self, level: int, latents):
        return self.sim.decode(level, latents, self.gt, self.ep)

    def _action_embedding(self, level: int, mode: str, action, gt_index: int):
        torch = self.sim.torch
        jepa = self.sim.jepas[level - 1]
        acts_gt = self.gt['act_embs'][level]
        if mode == 'ground_truth':
            i = min(gt_index, acts_gt.shape[1] - 1)
            return acts_gt[:, i:i + 1]
        if level == 1:
            a_dim = self.ep.actions.shape[1]
            if mode == 'user':
                raw = np.asarray(action, dtype=np.float32).reshape(1, a_dim)
            elif mode == 'zero':
                raw = np.zeros((1, a_dim), np.float32)
            elif mode == 'random':
                lo, hi = self.ep.actions.min(0), self.ep.actions.max(0)
                raw = np.random.uniform(lo, hi).astype(np.float32).reshape(1, a_dim)
            else:
                raise ValueError(f'unknown action mode {mode!r}')
            block = self.sim._stack_actions(np.repeat(raw, self.sim.frameskip, axis=0), 1)
            with torch.no_grad():
                return jepa.action_embed(torch.from_numpy(block).to(self.sim.device)[None])
        if mode in ('random', 'sample'):
            queue = jepa.get_latent_action_queue()
            if queue is None:
                raise ValueError('this model has no latent-action queue to sample from')
            row = queue[np.random.randint(len(queue))].reshape(1, 1, -1).to(self.sim.device)
            with torch.no_grad():
                return jepa.action_embed(row)
        raise ValueError(f'level {level} supports ground_truth or random macro-actions')

    def step(self, body: dict) -> dict:
        if self.ep is None:
            raise ValueError('call reset first')
        torch = self.sim.torch
        level = int(body.get('level', 1))
        if level not in self.state:
            raise ValueError(f'level must be 1..{self.sim.num_levels}')
        mode = body.get('mode', 'ground_truth')
        st = self.state[level]
        jepa = self.sim.jepas[level - 1]
        hs = self.sim.history[level - 1]
        with torch.no_grad():
            st['acts'].append(self._action_embedding(level, mode, body.get('action'), st['gt_pos']))
            states = st['latents'][:, -hs:]
            acts = torch.cat(st['acts'][-states.shape[1]:], dim=1)
            nxt = jepa.predict(states, acts)[:, -1:]
            st['latents'] = torch.cat([st['latents'], nxt], dim=1)
            st['gt_pos'] += 1
            st['dreamed'] += 1
            frame = self._frame(level, nxt[0])[0]
        gt_emb = self.gt['embeds'][level][0]
        fidx = self.gt['frame_index'][level]
        out = {'level': level, 'dreamed': st['dreamed'], 'frame': _b64png(frame), 'real': None,
               'latent_error': None, 'raw_step': None}
        if st['gt_pos'] < gt_emb.shape[0]:
            scale = gt_emb.var(dim=0).mean().clamp_min(1e-8)
            err = ((nxt[0, 0] - gt_emb[st['gt_pos']]) ** 2).mean() / scale
            out.update(real=_b64png(self.ep.pixels[fidx[st['gt_pos']]]),
                       latent_error=float(err), raw_step=int(fidx[st['gt_pos']]))
        return out

    def rollout(self, body: dict) -> dict:
        n = max(1, min(int(body.get('horizon', 8)), MAX_ROLLOUT))
        steps = [self.step(body) for _ in range(n)]
        return {'steps': steps}

    def sample_action(self, body: dict) -> dict:
        if self.ep is None:
            raise ValueError('call reset first')
        lo, hi = self.ep.actions.min(0), self.ep.actions.max(0)
        return {'action': np.random.uniform(lo, hi).round(4).tolist()}
