"""How the worker turns a platform job spec into H-JEPA inputs and commands.

Everything upstream-specific lives here: data download recipes for the official
datasets (README sections 3.1 and 3.2), training config generation for user data,
and the command lines for ``main_hjepa.py`` and ``eval.py``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

HJEPA_SRC = Path(os.environ.get('HJEPA_SRC', '/opt/H-JEPA'))
HJEPA_HOME = Path(os.environ.get('HJEPA_HOME', '/workspace/hjepa_home'))
H_JEPA_DIR = HJEPA_SRC / 'h_jepa'
HF_DATA_REPO = 'jepa-world-models/h-jepa'


def run(cmd: list[str], cwd: Path | None = None, log=print) -> None:
    log('$ ' + ' '.join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True, env={**os.environ, 'HJEPA_HOME': str(HJEPA_HOME)})


def hf_fetch(repo: str, patterns: list[str], local_dir: Path, repo_type: str = 'dataset',
             revision: str | None = None) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo, repo_type=repo_type, allow_patterns=patterns,
                      local_dir=str(local_dir), revision=revision)


# ----------------------------------------------------------------------------- data

def fetch_official_data(preset: str, log=print) -> tuple[str, str]:
    """Download (and preprocess) an official dataset; return (train_name, val_name)."""
    home = HJEPA_HOME
    home.mkdir(parents=True, exist_ok=True)
    if preset == 'pusht':
        if not (home / 'pusht_expert_train.h5').exists():
            hf_fetch('quentinll/lewm-pusht', ['*.zst'], home)
            run(['unzstd', '-f', str(home / 'pusht_expert_train.h5.zst')], log=log)
            (home / 'pusht_expert_train.h5.zst').unlink(missing_ok=True)
            run(['python', 'scripts/data/add_pusht_block_ori.py',
                 str(home / 'pusht_expert_train.h5')], cwd=HJEPA_SRC, log=log)
        if not (home / 'pusht_expert_val.h5').exists():
            hf_fetch(HF_DATA_REPO, ['pusht_expert_val.h5'], home)
        return 'pusht_expert_train', 'pusht_expert_val'
    if preset == 'cube':
        if not (home / 'cube_single_expert_train.h5').exists():
            hf_fetch('quentinll/lewm-cube', ['*.zst'], home)
            run(['tar', '-I', 'unzstd', '-xf', str(home / 'cube_single_expert.tar.zst'),
                 '-C', str(home)], log=log)
            (home / 'cube_single_expert.tar.zst').unlink(missing_ok=True)
            run(['python', 'scripts/data/split_cube.py'], cwd=HJEPA_SRC, log=log)
            (home / 'cube_single_expert.h5').unlink(missing_ok=True)
        return 'cube_single_expert_train', 'cube_single_expert_val'
    if preset == 'ant':
        hf_fetch(HF_DATA_REPO, ['visual_antmaze_medium_*'], home)
        return 'visual_antmaze_medium_explore_stitch_train', 'visual_antmaze_medium_stitch_val'
    if preset == 'fourroom':
        hf_fetch(HF_DATA_REPO, ['fourroom_tp35_d1*'], home)
        return 'fourroom_tp35_d1', 'fourroom_tp35_d1_val'
    raise ValueError(f'no official data for preset {preset!r}')


def fetch_eval_tasks(log=print) -> None:
    target = H_JEPA_DIR / 'assets'
    if not (target / 'eval_trajs').exists():
        log('downloading the 50 official start/goal evaluation tasks')
        hf_fetch(HF_DATA_REPO, ['eval_trajs/*'], target)


def split_h5(src: Path, train_dst: Path, val_dst: Path, val_frac: float = 0.05) -> tuple[int, int]:
    """Split an episode HDF5 file by episodes (last val_frac go to validation)."""
    import h5py

    try:
        import hdf5plugin  # noqa: F401
    except ImportError:
        pass

    with h5py.File(src, 'r') as f:
        ep_len, ep_offset = f['ep_len'][:], f['ep_offset'][:]
        n = len(ep_len)
        n_val = max(1, int(round(n * val_frac))) if n > 1 else 0
        if n_val == 0:
            raise ValueError('need at least 2 episodes to make a validation split')
        splits = {train_dst: (0, n - n_val), val_dst: (n - n_val, n)}
        for dst_path, (e0, e1) in splits.items():
            r0 = int(ep_offset[e0])
            r1 = int(ep_offset[e1 - 1] + ep_len[e1 - 1])
            with h5py.File(dst_path, 'w') as dst:
                dst['ep_len'] = ep_len[e0:e1]
                dst['ep_offset'] = (ep_offset[e0:e1] - r0).astype(ep_offset.dtype)
                for key, ds in f.items():
                    if key in ('ep_len', 'ep_offset') or not isinstance(ds, h5py.Dataset):
                        continue
                    if ds.shape and ds.shape[0] >= r1:
                        out = dst.create_dataset(key, shape=(r1 - r0, *ds.shape[1:]),
                                                 dtype=ds.dtype, chunks=ds.chunks)
                        step = 4096
                        for i in range(r0, r1, step):
                            block = ds[i:min(i + step, r1)]
                            out[i - r0:i - r0 + len(block)] = block
                    else:
                        dst[key] = ds[()]
        return n - n_val, n_val


def h5_shape_info(path: Path) -> dict:
    import h5py

    with h5py.File(path, 'r') as f:
        return {'img_size': int(f['pixels'].shape[1]), 'action_dim': int(f['action'].shape[1]),
                'has_proprio': 'proprio' in f}


# ----------------------------------------------------------------------------- configs

def _patch_size_for(img_size: int) -> int:
    if img_size % 16 == 0 and img_size // 16 >= 2:
        return img_size // 16
    for p in (16, 14, 8, 7, 4, 2):
        if img_size % p == 0 and img_size // p <= 32:
            return p
    raise ValueError(f'cannot pick a ViT patch size for {img_size}px frames')


def write_custom_configs(variant: str, train_name: str, val_name: str, img_size: int,
                         action_dim: int, frameskip: int, levels: int) -> str:
    """Generate config/train/custom_<variant>.yaml (+ base and probing) from the Push-T
    configs, adapted to a (pixels, action) dataset. Returns the config name."""
    import yaml

    cfg_dir = H_JEPA_DIR / 'config'
    base = yaml.safe_load((cfg_dir / 'train/base/pusht.yaml').read_text())
    steps = "${eval:'${level1.wm.history_size} + ${level1.wm.rollout_n}'}"
    base['data']['dataset'] = {
        'name': train_name, 'val_name': val_name,
        'keys_to_load': ['pixels', 'action'], 'keys_to_cache': ['action'],
        'level1': {'frameskip': frameskip, 'num_steps': steps},
    }
    base['env'] = 'custom'
    base['img_size'] = img_size
    base['patch_size'] = _patch_size_for(img_size)
    base['level1']['wm']['action_dim'] = action_dim
    base['level1']['wm']['use_proprio'] = False
    base['planning_eval']['enabled'] = False
    base['final_probing_decoding_eval']['config_name'] = 'custom'
    (cfg_dir / 'train/base/custom.yaml').write_text(
        '# @package _global_\n# Generated by hjepa_worker for user data.\n'
        + yaml.safe_dump(base, sort_keys=False))

    model = yaml.safe_load((cfg_dir / f'train/pusht_{variant}.yaml').read_text())
    model['defaults'] = ['base/custom', '_self_']
    model['output_model_name'] = f'custom_{variant}'
    model.setdefault('planning_eval', {})['config_name'] = 'pusht_flat'
    overrides = model.setdefault('final_probing_decoding_eval', {}).setdefault('overrides', {})
    for level in range(1, levels + 1):
        overrides[f'level{level}'] = {'train_decoder': True, 'probes': {'enabled': False}}
    (cfg_dir / f'train/custom_{variant}.yaml').write_text(yaml.safe_dump(model, sort_keys=False))

    probing = yaml.safe_load((cfg_dir / 'probing/pusht.yaml').read_text())
    for key, name in (('train_dataset', train_name), ('eval_dataset', val_name)):
        probing[key] = {'name': name, 'keys_to_load': ['pixels', 'action'],
                        'keys_to_cache': ['action'], 'level1': {'frameskip': 1, 'num_steps': 1}}
    (cfg_dir / 'probing/custom.yaml').write_text(yaml.safe_dump(probing, sort_keys=False))
    return f'custom_{variant}'


def train_command(params: dict, config_name: str, train_name: str, val_name: str,
                  official: bool, levels: int) -> list[str]:
    workers = max(2, min(12, (os.cpu_count() or 4) - 2))
    cmd = [
        'python', 'main_hjepa.py', f'--config-name={config_name}',
        f'seed={params["seed"]}',
        f'trainer.max_epochs={params["epochs"]}',
        f'loader.batch_size={params["batch_size"]}',
        f'num_workers={workers}',
        f'trainer.devices={params.get("gpu_count", 1)}',
        'wandb.enabled=false',
        f'planning_eval.enabled={str(bool(params.get("run_planning_eval"))).lower()}',
        f'final_probing_decoding_eval.enabled={str(bool(params.get("train_decoders", True))).lower()}',
    ]
    if params['preset'] != 'custom' and not official:
        # A built-in env trained on user data: point training and probing at it.
        cmd += [
            f'data.dataset.name={train_name}',
            f'data.dataset.val_name={val_name}',
            '++data.dataset.val_total_transitions=null',
            f'++final_probing_decoding_eval.overrides.train_dataset.name={train_name}',
            '++final_probing_decoding_eval.overrides.train_dataset.total_transitions=null',
            f'++final_probing_decoding_eval.overrides.eval_dataset.name={val_name}',
        ]
    cmd += list(params.get('extra_overrides') or [])
    return cmd


def run_dir_for(params: dict, config_name: str) -> Path:
    env = 'custom' if params['preset'] == 'custom' else params['preset']
    return HJEPA_HOME / 'ckpts' / env / config_name / f'seed{params["seed"]}'


def eval_command(env: str, planner: str, ckpt: Path, seed: int, num_eval: int) -> list[str]:
    return ['python', 'eval.py', f'--config-name={env}_{planner}', f'seed={seed}',
            'output.dir=platform_eval', f'policy={ckpt}', f'eval.num_eval={num_eval}']
