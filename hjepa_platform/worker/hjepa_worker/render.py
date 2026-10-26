"""Turn dreamed frames into browser-friendly media (GIF comparisons, PNG grids)."""

from __future__ import annotations

import io

import numpy as np

TILE = 160


def _tile(frame: np.ndarray | None):
    from PIL import Image

    if frame is None:
        return Image.new('RGB', (TILE, TILE), (24, 24, 28))
    return Image.fromarray(frame).resize((TILE, TILE), Image.NEAREST)


def comparison_gif(context: np.ndarray, real: np.ndarray, pred: np.ndarray, label: str,
                   fps: int = 6) -> bytes:
    """Side-by-side [real | dreamed] animation; context frames show on both sides."""
    from PIL import Image, ImageDraw

    frames = []
    seq = [(c, c, 'context') for c in context]
    seq += [(real[i] if i < len(real) else None, pred[i], f'{label} t+{i + 1}')
            for i in range(len(pred))]
    for left, right, caption in seq:
        canvas = Image.new('RGB', (TILE * 2 + 6, TILE + 22), (12, 12, 16))
        canvas.paste(_tile(left), (0, 22))
        canvas.paste(_tile(right), (TILE + 6, 22))
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 4), 'real', fill=(200, 200, 200))
        draw.text((TILE + 10, 4), f'dream · {caption}', fill=(140, 200, 255))
        frames.append(canvas)
    buf = io.BytesIO()
    frames[0].save(buf, format='GIF', save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0)
    return buf.getvalue()


def comparison_grid(rows: list[tuple[str, list[np.ndarray | None]]], max_cols: int = 12) -> bytes:
    """Labelled grid, one row per (label, frames)."""
    from PIL import Image, ImageDraw

    label_w = 110
    small = TILE // 2
    n_cols = min(max_cols, max((len(f) for _, f in rows), default=1))
    canvas = Image.new('RGB', (label_w + n_cols * (small + 2), len(rows) * (small + 2)), (12, 12, 16))
    draw = ImageDraw.Draw(canvas)
    for r, (label, frames) in enumerate(rows):
        y = r * (small + 2)
        draw.text((6, y + small // 2 - 6), label, fill=(220, 220, 220))
        idx = np.linspace(0, len(frames) - 1, num=min(n_cols, len(frames))).astype(int) if frames else []
        for c, i in enumerate(idx):
            canvas.paste(_tile(frames[i]).resize((small, small)), (label_w + c * (small + 2), y))
    buf = io.BytesIO()
    canvas.save(buf, format='PNG')
    return buf.getvalue()
