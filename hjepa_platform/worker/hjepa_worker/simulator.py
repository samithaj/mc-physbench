"""Latent "dreaming" with a trained H-JEPA: encode real frames at every level, roll each
level's predictor forward under chosen actions, and decode predictions back to pixels.

Decoding uses the CLS decoders trained by upstream's final_probing_decoding_eval
(``decoder_level{N}.ckpt``). Without them, a prediction is shown as its nearest real
frame in the episode (latent nearest-neighbour retrieval).

Model internals follow h_jepa/ARCHITECTURE.md: level 1 encodes pixels (plus proprio when
the encoder is a fusion encoder); level N encodes chunks of level N-1 embeddings and
compresses chunks of level N-1 action embeddings into a macro-action.
"""

from __future__ import annotations

import io
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .recipes import H_JEPA_DIR, HJEPA_SRC

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ACTION_MODES = ('ground_truth', 'zero', 'random', 'reversed')


def _import_hjepa():
    for p in (str(H_JEPA_DIR), str(HJEPA_SRC)):
        if p not in sys.path:
            sys.path.insert(0, p)


@dataclass
class Episode:
    pixels: np.ndarray          # (T, H, W, 3) uint8, every raw step
    actions: np.ndarray         # (T, A) raw actions
    proprio: np.ndarray | None  # (T, P)


def load_episode(h5_path: Path, index: int) -> Episode:
    import h5py

    try:
        import hdf5plugin  # noqa: F401
    except ImportError:
        pass
    with h5py.File(h5_path, 'r') as f:
        n = len(f['ep_len'])
        if not 0 <= index < n:
            raise ValueError(f'episode {index} out of range (dataset has {n})')
        s = int(f['ep_offset'][index])
        e = s + int(f['ep_len'][index])
        pixels = f['pixels'][s:e]
        actions = f['action'][s:e].astype(np.float32)
        proprio = None
        if 'proprio' in f:
            proprio = f['proprio'][s:e].astype(np.float32)
        elif 'qpos' in f and 'qvel' in f:  # Visual AntMaze merges qpos[2:] and qvel
            proprio = np.concatenate([f['qpos'][s:e][:, 2:], f['qvel'][s:e]], axis=1).astype(np.float32)
    return Episode(pixels, actions, proprio)


def episode_count(h5_path: Path) -> int:
    import h5py

    with h5py.File(h5_path, 'r') as f:
        return int(len(f['ep_len']))


def png_bytes(frame: np.ndarray, size: int | None = None) -> bytes:
    from PIL import Image

    img = Image.fromarray(frame)
    if size and img.size[0] != size:
        img = img.resize((size, size), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


class WorldModelSimulator:
    def __init__(self, ckpt: Path, normalizer: Path, config: Path | None,
                 decoders: dict[int, Path], device: str | None = None):
        _import_hjepa()
        import torch
        from omegaconf import OmegaConf

        self.torch = torch
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        model = torch.load(ckpt, map_location='cpu', weights_only=False)
        self.jepas = list(model.jepas) if hasattr(model, 'jepas') else [model]
        self.model = model.to(self.device).eval()
        self.num_levels = len(self.jepas)
        self.normalizer = torch.load(normalizer, map_location='cpu', weights_only=False)['stats']

        cfg = OmegaConf.load(config) if config and Path(config).exists() else OmegaConf.create({})
        self.frameskip = int(OmegaConf.select(cfg, 'data.dataset.level1.frameskip', default=5))
        self.img_size = OmegaConf.select(cfg, 'img_size', default=None)
        self.history = [int(OmegaConf.select(cfg, f'level{lv}.wm.history_size', default=3))
                        for lv in range(1, self.num_levels + 1)]
        self.uses_proprio = hasattr(self.jepas[0].encoder, 'proprio_encoder')
        for lv in range(2, self.num_levels + 1):
            if hasattr(self.jepas[lv - 1].encoder, 'proprio_encoder'):
                raise NotImplementedError('upper levels with proprio fusion are not supported')

        self.decoders = {}
        if decoders:
            from models.module import CLSDecoder
        for level, path in decoders.items():
            payload = torch.load(path, map_location='cpu', weights_only=False)
            dec = CLSDecoder(cls_dim=payload['cls_dim'], img_size=payload['img_size'],
                             patch_size=payload['patch_size'], **payload['decoder_kwargs'])
            dec.load_state_dict(payload['decoder_state_dict'])
            self.decoders[int(level)] = dec.to(self.device).eval()

    # ------------------------------------------------------------------ normalization

    def _norm(self, col: str, x: np.ndarray) -> np.ndarray:
        stats = self.normalizer.get(col)
        if stats is None:
            return x
        out = (x - np.asarray(stats['mean'], dtype=np.float32)) / np.asarray(stats['std'], dtype=np.float32)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _pixels_tensor(self, frames: np.ndarray):
        torch = self.torch
        x = torch.from_numpy(np.ascontiguousarray(frames)).to(self.device).permute(0, 3, 1, 2).float() / 255
        if self.img_size and x.shape[-1] != int(self.img_size):
            x = torch.nn.functional.interpolate(x, size=(int(self.img_size),) * 2, mode='bilinear',
                                                align_corners=False)
        mean = x.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = x.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
        return ((x - mean) / std).unsqueeze(0)  # (1, T, 3, H, W)

    def _to_uint8(self, images) -> np.ndarray:
        mean = images.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = images.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
        x = (images * std + mean).clamp(0, 1)
        return (x.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

    # ------------------------------------------------------------------ actions

    def level1_actions(self, ep: Episode, mode: str = 'ground_truth', scale: float = 1.0,
                       seed: int = 0) -> np.ndarray:
        """Raw per-step actions after applying an intervention, (T, A)."""
        a = ep.actions.copy()
        if mode == 'zero':
            a[:] = 0
        elif mode == 'random':
            rng = np.random.default_rng(seed)
            lo, hi = ep.actions.min(0), ep.actions.max(0)
            a = rng.uniform(lo, hi, size=a.shape).astype(np.float32)
        elif mode == 'reversed':
            a = -a
        elif mode != 'ground_truth':
            raise ValueError(f'unknown action mode {mode!r}')
        return (a * scale).astype(np.float32)

    def _stack_actions(self, raw: np.ndarray, n_frames: int) -> np.ndarray:
        """Group raw actions into level-1 blocks of `frameskip`, normalized, (n, fs*A)."""
        fs = self.frameskip
        need = n_frames * fs
        if raw.shape[0] < need:
            raw = np.concatenate([raw, np.zeros((need - raw.shape[0], raw.shape[1]), raw.dtype)])
        blocks = self._norm('action', raw[:need]).reshape(n_frames, fs, -1)
        return blocks.reshape(n_frames, -1)

    # ------------------------------------------------------------------ encoding

    def encode(self, ep: Episode, raw_actions: np.ndarray | None = None) -> dict:
        """Encode an episode at every level: embeds, action embeddings, and for each level
        the raw-step index of the frame each latent ends on."""
        torch = self.torch
        fs = self.frameskip
        n1 = (len(ep.pixels) - 1) // fs + 1
        frames = ep.pixels[::fs][:n1]
        raw_actions = ep.actions if raw_actions is None else raw_actions
        act1 = torch.from_numpy(self._stack_actions(raw_actions, n1)).to(self.device)[None]
        out: dict = {'n1': n1}
        with torch.no_grad():
            chunks = []
            for i in range(0, n1, 64):
                info = {'pixels': self._pixels_tensor(frames[i:i + 64])}
                if self.uses_proprio and ep.proprio is not None:
                    prop = self._norm('proprio', ep.proprio[::fs][:n1][i:i + 64])
                    info['proprio'] = torch.from_numpy(prop).to(self.device)[None]
                chunks.append(self.jepas[0].encode(info)['embed_0'])
            embeds = {1: torch.cat(chunks, dim=1)}
            act_embs = {1: self.jepas[0].action_embed(act1)}
            frame_index = {1: np.arange(n1) * fs}
            for lv in range(2, self.num_levels + 1):
                jepa = self.jepas[lv - 1]
                key = f'embed_{lv - 1}'
                enc = jepa.encode({key: embeds[lv - 1], 'action': act_embs[lv - 1]},
                                  key=key, chunk_temporal_inputs=True)
                embeds[lv] = enc['embed_0']
                act_embs[lv] = enc['action_0'][:, :embeds[lv].shape[1]]
                stride, win = jepa.temporal_stride, jepa.temporal_window_size
                n = embeds[lv].shape[1]
                lower = frame_index[lv - 1]
                frame_index[lv] = np.array([lower[min(j * stride + win - 1, len(lower) - 1)]
                                            for j in range(n)])
        out.update(embeds=embeds, act_embs=act_embs, frame_index=frame_index)
        return out

    # ------------------------------------------------------------------ rollout

    def rollout(self, encoded: dict, level: int, start: int, context: int, horizon: int,
                act_embs=None):
        """Autoregressive prediction at `level` from `context` real latents starting at
        `start`, for `horizon` steps. Returns predicted latents (horizon, D)."""
        torch = self.torch
        jepa = self.jepas[level - 1]
        gt = encoded['embeds'][level]
        acts = encoded['act_embs'][level] if act_embs is None else act_embs
        n = gt.shape[1]
        context = max(1, min(context, n - start))
        horizon = max(0, min(horizon, acts.shape[1] - (start + context) + 1))
        hs = self.history[level - 1]
        states = gt[:, start:start + context]
        preds = []
        with torch.no_grad():
            for t in range(horizon):
                k = states.shape[1]
                a = acts[:, start + max(0, k - hs):start + k]
                nxt = jepa.predict(states[:, -hs:], a)[:, -1:]
                states = torch.cat([states, nxt], dim=1)
                preds.append(nxt)
        if not preds:
            return gt.new_zeros((0, gt.shape[-1])), context
        return torch.cat(preds, dim=1)[0], context

    def decode(self, level: int, latents, encoded: dict, ep: Episode) -> np.ndarray:
        """Latents (N, D) at `level` -> uint8 frames (N, H, W, 3)."""
        torch = self.torch
        if len(latents) == 0:
            return np.zeros((0, *ep.pixels.shape[1:]), np.uint8)
        with torch.no_grad():
            if level in self.decoders:
                return self._to_uint8(self.decoders[level](latents))
            bank = encoded['embeds'][level][0]
            idx = torch.cdist(latents, bank).argmin(dim=1).cpu().numpy()
        return ep.pixels[encoded['frame_index'][level][idx]]

    def latent_errors(self, level: int, preds, encoded: dict, start: int, context: int) -> list[float]:
        gt = encoded['embeds'][level][0]
        target = gt[start + context:start + context + len(preds)]
        m = min(len(target), len(preds))
        if m == 0:
            return []
        scale = gt.var(dim=0).mean().clamp_min(1e-8)
        err = ((preds[:m] - target[:m]) ** 2).mean(dim=1) / scale
        return [float(v) for v in err.cpu()]

    def dream(self, ep: Episode, levels: list[int], start_frame: int, context: int, horizon: int,
              action_mode: str = 'ground_truth', action_scale: float = 1.0, seed: int = 0) -> dict:
        """Roll out every requested level from the same raw start frame. Level-N horizons
        and contexts are in level-N steps."""
        gt_enc = self.encode(ep)
        raw = self.level1_actions(ep, action_mode, action_scale, seed)
        act_enc = gt_enc if action_mode == 'ground_truth' and action_scale == 1.0 else self.encode(ep, raw)
        out = {}
        for lv in levels:
            fidx = gt_enc['frame_index'][lv]
            start = int(np.searchsorted(fidx, start_frame))
            start = min(start, max(0, len(fidx) - 2))
            preds, ctx = self.rollout(gt_enc, lv, start, context, horizon, act_enc['act_embs'][lv])
            pred_frames = self.decode(lv, preds, gt_enc, ep)
            first = start + ctx
            real_idx = fidx[first:first + len(preds)]
            real_frames = ep.pixels[real_idx] if len(real_idx) else np.zeros_like(pred_frames)
            ctx_frames = ep.pixels[fidx[start:start + ctx]]
            out[lv] = {
                'context_frames': ctx_frames,
                'pred_frames': pred_frames,
                'real_frames': real_frames,
                'raw_steps': [int(i) for i in fidx[first:first + len(preds)]],
                'latent_error': self.latent_errors(lv, preds, gt_enc, start, ctx),
                'pixel_error': ([float(np.mean((p.astype(np.float32) - r.astype(np.float32)) ** 2) / 255 ** 2)
                                 for p, r in zip(pred_frames, real_frames)]
                                if lv in self.decoders else []),
                'decoded_with': 'decoder' if lv in self.decoders else 'nearest_neighbour',
            }
        return out
