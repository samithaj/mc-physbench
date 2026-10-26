"""Static catalog: H-JEPA environments, model variants, planners and rentable GPUs.

Names mirror the upstream repo (https://github.com/kevinghst/H-JEPA): training configs are
``config/train/<env>_<variant>.yaml`` and planners ``config/eval/<env>_<planner>.yaml``.
"""

from __future__ import annotations

HJEPA_REPO = 'https://github.com/kevinghst/H-JEPA'
HJEPA_COMMIT = '97054ecaff83e13839195ec4b92d74c08a0127c5'
HF_REPO = 'jepa-world-models/h-jepa'

ENV_PRESETS: dict[str, dict] = {
    'pusht': {
        'label': 'Push-T',
        'description': '2D pushing: a circular agent pushes a T-block to a target pose. 224px frames.',
        'img_size': 224,
        'action_dim': 2,
        'frameskip': 5,
        'required_columns': ['pixels', 'action', 'proprio', 'state'],
        'official': {'train': 'pusht_expert_train', 'val': 'pusht_expert_val',
                     'source': 'quentinll/lewm-pusht (MIT) + jepa-world-models/h-jepa'},
        'simulator': True,
    },
    'ant': {
        'label': 'Visual AntMaze',
        'description': 'OGBench ant navigating a medium maze from 64px top-down frames.',
        'img_size': 64,
        'action_dim': 8,
        'frameskip': 5,
        'required_columns': ['pixels', 'action', 'xy', 'qpos', 'qvel'],
        'official': {'train': 'visual_antmaze_medium_explore_stitch_train',
                     'val': 'visual_antmaze_medium_stitch_val',
                     'source': 'jepa-world-models/h-jepa'},
        'simulator': True,
    },
    'fourroom': {
        'label': 'FourRoom Distractors',
        'description': 'Navigation across four rooms with moving distractors. 224px frames.',
        'img_size': 224,
        'action_dim': 2,
        'frameskip': 5,
        'required_columns': ['pixels', 'action'],
        'official': {'train': 'fourroom_tp35_d1', 'val': 'fourroom_tp35_d1_val',
                     'source': 'jepa-world-models/h-jepa'},
        'simulator': True,
    },
    'cube': {
        'label': 'OGBench Cube',
        'description': 'Robot arm single-cube pick and place (OGBench).',
        'img_size': 224,
        'action_dim': 5,
        'frameskip': 5,
        'required_columns': ['pixels', 'action'],
        'official': {'train': 'cube_single_expert_train', 'val': 'cube_single_expert_val',
                     'source': 'quentinll/lewm-cube (MIT), split 9,800 / 200 episodes'},
        'simulator': True,
    },
    'custom': {
        'label': 'Custom data',
        'description': ('Any (pixels, action) video dataset. Architecture follows the Push-T '
                        'configs; image size, patch size and action dim come from the data. '
                        'No simulator, so planning eval is off; decoders are still trained.'),
        'img_size': None,
        'action_dim': None,
        'frameskip': 5,
        'required_columns': ['pixels', 'action'],
        'official': None,
        'simulator': False,
    },
}

VARIANTS: dict[str, dict] = {
    'lewm': {'label': 'LeWM (flat, 1 level)', 'levels': 1, 'family': 'flat'},
    'hjepa_l2': {'label': 'H-JEPA, 2 levels', 'levels': 2, 'family': 'hjepa'},
    'hjepa_l3': {'label': 'H-JEPA, 3 levels', 'levels': 3, 'family': 'hjepa'},
    'hjepa_l4': {'label': 'H-JEPA, 4 levels', 'levels': 4, 'family': 'hjepa'},
    'hwm_l2': {'label': 'HWM, 2 levels (identity upper encoders)', 'levels': 2, 'family': 'hwm'},
    'hwm_l3': {'label': 'HWM, 3 levels', 'levels': 3, 'family': 'hwm'},
    'hwm_l4': {'label': 'HWM, 4 levels', 'levels': 4, 'family': 'hwm'},
}


def planners_for(levels: int) -> list[str]:
    """Planners valid for a model with `levels` levels (eval config suffixes)."""
    out = ['flat']
    for k in range(2, levels + 1):
        out += [f'l{k}', f'l{k}_project']
    return out


def default_planner(levels: int) -> str:
    return 'flat' if levels == 1 else f'l{levels}'


# Small-scale GPUs rentable on RunPod. ids are RunPod gpuTypeIds; prices are fallbacks
# shown when live pricing is unavailable (refreshed from the RunPod API when a key is set).
GPU_CATALOG: list[dict] = [
    {'id': 'NVIDIA GeForce RTX 5090', 'label': 'RTX 5090', 'vram_gb': 32, 'est_usd_hr': 0.89,
     'arch': 'blackwell', 'recommended': True},
    {'id': 'NVIDIA GeForce RTX 4090', 'label': 'RTX 4090', 'vram_gb': 24, 'est_usd_hr': 0.59,
     'arch': 'ada'},
    {'id': 'NVIDIA GeForce RTX 3090', 'label': 'RTX 3090', 'vram_gb': 24, 'est_usd_hr': 0.43,
     'arch': 'ampere'},
    {'id': 'NVIDIA RTX A5000', 'label': 'RTX A5000', 'vram_gb': 24, 'est_usd_hr': 0.27,
     'arch': 'ampere'},
    {'id': 'NVIDIA L4', 'label': 'L4', 'vram_gb': 24, 'est_usd_hr': 0.43, 'arch': 'ada'},
    {'id': 'NVIDIA RTX 4000 Ada Generation', 'label': 'RTX 4000 Ada', 'vram_gb': 20,
     'est_usd_hr': 0.26, 'arch': 'ada'},
    {'id': 'NVIDIA RTX A4000', 'label': 'RTX A4000', 'vram_gb': 16, 'est_usd_hr': 0.25,
     'arch': 'ampere'},
]
GPU_IDS = {g['id'] for g in GPU_CATALOG}
DEFAULT_GPU = 'NVIDIA GeForce RTX 5090'

# Rough wall-clock hours for one training run on a single RTX 5090-class GPU,
# used only for the cost estimate shown before launch.
EST_TRAIN_HOURS_PER_EPOCH = {'pusht': 0.35, 'ant': 0.25, 'fourroom': 0.08, 'cube': 0.35,
                             'custom': 0.25}


def gpu_entry(gpu_id: str) -> dict | None:
    return next((g for g in GPU_CATALOG if g['id'] == gpu_id), None)


def official_model_catalog() -> list[dict]:
    """The 84 released checkpoints, exposed as seed-42 entries per env x variant."""
    out = []
    for env in ('pusht', 'ant', 'fourroom', 'cube'):
        for variant, meta in VARIANTS.items():
            out.append({
                'env': env,
                'variant': variant,
                'levels': meta['levels'],
                'name': f'{ENV_PRESETS[env]["label"]} - {meta["label"]} (official, seed 42)',
                'hf_prefix': f'{env}/{env}_{variant}/seed42',
                'object_ckpt': f'{env}_{variant}_object.ckpt',
            })
    return out
