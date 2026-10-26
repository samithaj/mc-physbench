"""Validate an uploaded HDF5 file against H-JEPA's dataset layout and render a preview.

The layout read by ``stable_worldmodel.data.HDF5Dataset``: every column is a flat array
over all steps of all episodes, indexed by ``ep_offset[i] : ep_offset[i] + ep_len[i]``.
``pixels`` is uint8 (N, H, W, 3) and ``action`` is float (N, action_dim).
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np

try:  # official files use Blosc/Zstd filters registered by hdf5plugin
    import hdf5plugin  # noqa: F401
except ImportError:  # pragma: no cover
    pass

from .presets import ENV_PRESETS


def _column_info(ds) -> dict:
    return {'shape': list(ds.shape), 'dtype': str(ds.dtype),
            'compression': ds.compression or None}


def inspect_h5(path: str | Path, preview_frames: int = 8) -> tuple[dict, bytes | None]:
    """Return (report, preview_png). report['valid'] is False with report['errors'] set
    when the file cannot be trained on."""
    import h5py

    errors: list[str] = []
    warnings: list[str] = []
    report: dict = {'valid': False, 'errors': errors, 'warnings': warnings}
    try:
        f = h5py.File(path, 'r')
    except Exception as exc:  # not HDF5, truncated, ...
        errors.append(f'not a readable HDF5 file: {exc}')
        return report, None

    with f:
        columns = {k: _column_info(f[k]) for k in f.keys() if isinstance(f[k], h5py.Dataset)}
        report['columns'] = columns
        for key in ('ep_len', 'ep_offset', 'pixels', 'action'):
            if key not in columns:
                errors.append(f"missing required column '{key}'")
        if errors:
            return report, None

        ep_len = f['ep_len'][:].astype(np.int64)
        ep_offset = f['ep_offset'][:].astype(np.int64)
        n_steps = int(f['pixels'].shape[0])
        pixels_shape = f['pixels'].shape
        action_shape = f['action'].shape

        if ep_len.shape != ep_offset.shape:
            errors.append('ep_len and ep_offset have different lengths')
        elif len(ep_len) == 0:
            errors.append('dataset has no episodes')
        elif int((ep_offset + ep_len).max()) > n_steps:
            errors.append('episode ranges run past the end of pixels')
        if len(pixels_shape) != 4 or pixels_shape[-1] != 3:
            errors.append(f'pixels must be (N, H, W, 3), got {list(pixels_shape)}')
        elif pixels_shape[1] != pixels_shape[2]:
            errors.append(f'frames must be square, got {pixels_shape[1]}x{pixels_shape[2]}')
        if f['pixels'].dtype != np.uint8:
            errors.append(f"pixels must be uint8, got {f['pixels'].dtype}")
        if len(action_shape) != 2:
            errors.append(f'action must be (N, action_dim), got {list(action_shape)}')
        elif action_shape[0] != n_steps:
            errors.append('action and pixels have different numbers of rows')

        for name, info in columns.items():
            if name in ('ep_len', 'ep_offset') or not info['shape']:
                continue
            if info['shape'][0] != n_steps:
                warnings.append(f"column '{name}' has {info['shape'][0]} rows, expected {n_steps}")

        report.update(
            episodes=int(len(ep_len)),
            steps=n_steps,
            min_episode_len=int(ep_len.min()) if len(ep_len) else 0,
            max_episode_len=int(ep_len.max()) if len(ep_len) else 0,
            mean_episode_len=float(ep_len.mean()) if len(ep_len) else 0.0,
            img_size=int(pixels_shape[1]) if len(pixels_shape) == 4 else None,
            action_dim=int(action_shape[1]) if len(action_shape) == 2 else None,
            has_proprio='proprio' in columns,
        )
        if errors:
            return report, None

        report['compatible_presets'] = [
            key for key, preset in ENV_PRESETS.items()
            if all(c in columns for c in preset['required_columns'])
            and (preset['img_size'] in (None, report['img_size']))
            and (preset['action_dim'] in (None, report['action_dim']))
        ]
        if report['min_episode_len'] < 20:
            warnings.append('some episodes are shorter than 20 steps; with frameskip 5 and '
                            'deeper hierarchies they will be skipped')

        preview = None
        try:
            preview = _preview_strip(f['pixels'], int(ep_offset[0]), int(ep_len[0]),
                                     preview_frames)
        except Exception as exc:  # compressed with a filter we can't decode, ...
            warnings.append(f'could not render preview: {exc}')

    report['valid'] = True
    return report, preview


def _preview_strip(pixels, start: int, length: int, n: int) -> bytes:
    from PIL import Image

    idx = np.linspace(start, start + max(length - 1, 0), num=min(n, max(length, 1))).astype(int)
    frames = [np.asarray(pixels[int(i)]) for i in idx]
    tile = 128
    strip = Image.new('RGB', (tile * len(frames), tile))
    for k, frame in enumerate(frames):
        img = Image.fromarray(frame).resize((tile, tile), Image.NEAREST)
        strip.paste(img, (k * tile, 0))
    buf = io.BytesIO()
    strip.save(buf, format='PNG')
    return buf.getvalue()
