"""Run the worker's simulator against a real (randomly initialized) H-JEPA built from the
upstream configs. Skipped unless torch, hydra and an H-JEPA checkout are available:

    HJEPA_SRC=/path/to/H-JEPA pytest tests/test_simulator_real.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

HJEPA_SRC = Path(os.environ.get('HJEPA_SRC', '/opt/H-JEPA'))
torch = pytest.importorskip('torch')
pytest.importorskip('hydra')
pytest.importorskip('transformers')
if not (HJEPA_SRC / 'h_jepa' / 'main_hjepa.py').exists():
    pytest.skip('H-JEPA checkout not found (set HJEPA_SRC)', allow_module_level=True)

IMG = 28  # 2x2 patches of 14px keeps the ViT tiny on CPU


def _build(tmp_path, config_name, overrides=()):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    sys.path.insert(0, str(HJEPA_SRC / 'h_jepa'))
    sys.path.insert(0, str(HJEPA_SRC))
    if not OmegaConf.has_resolver('eval'):
        OmegaConf.register_new_resolver('eval', eval)
    with initialize_config_dir(config_dir=str(HJEPA_SRC / 'h_jepa/config/train'), version_base=None):
        cfg = compose(config_name=config_name,
                      overrides=[f'img_size={IMG}', 'patch_size=14', *overrides])
    from hjepa_utils import create_world_model
    from models.module import CLSDecoder

    model, _ = create_world_model(cfg)
    ckpt = tmp_path / 'm_object.ckpt'
    torch.save(model, ckpt)
    act_dim = int(cfg.level1.wm.action_dim)
    torch.save({'format': 'lejepa_training_normalizer_v1',
                'stats': {'action': {'mean': np.zeros(act_dim, np.float32),
                                     'std': np.ones(act_dim, np.float32), 'count': 1}}},
               tmp_path / 'normalizer.pt')
    OmegaConf.save(cfg, tmp_path / 'config.yaml')
    dec_dir = tmp_path / 'final_probing_decoding_eval'
    dec_dir.mkdir()
    dec = CLSDecoder(cls_dim=int(cfg.level1.wm.embed_dim), img_size=IMG, patch_size=14,
                     dim=32, heads=4, depth=1)
    torch.save({'format': 'final_probing_decoding_eval_decoder_v1', 'level': 1,
                'decoder_state_dict': dec.state_dict(),
                'decoder_kwargs': {'dim': 32, 'heads': 4, 'depth': 1},
                'cls_dim': int(cfg.level1.wm.embed_dim), 'img_size': IMG, 'patch_size': 14},
               dec_dir / 'decoder_level1.ckpt')
    return ckpt, act_dim, int(cfg.num_levels)


def _h5(path, act_dim, episodes=3, length=120):
    import h5py

    n = episodes * length
    with h5py.File(path, 'w') as f:
        f['ep_len'] = np.full(episodes, length, np.int64)
        f['ep_offset'] = np.arange(episodes, dtype=np.int64) * length
        f['pixels'] = np.random.randint(0, 255, (n, IMG, IMG, 3), np.uint8)
        f['action'] = np.random.randn(n, act_dim).astype(np.float32)


@pytest.mark.parametrize('config_name', ['pusht_hjepa_l2', 'pusht_hjepa_l3', 'pusht_lewm'])
def test_dream_and_interactive(tmp_path, config_name):
    os.environ['HJEPA_SRC'] = str(HJEPA_SRC)
    from hjepa_worker.engine import InteractiveEngine
    from hjepa_worker.simulator import WorldModelSimulator, load_episode

    ckpt, act_dim, levels = _build(tmp_path, config_name)
    h5 = tmp_path / 'data.h5'
    _h5(h5, act_dim)
    sim = WorldModelSimulator(ckpt, tmp_path / 'normalizer.pt', tmp_path / 'config.yaml',
                              {1: tmp_path / 'final_probing_decoding_eval/decoder_level1.ckpt'},
                              device='cpu')
    assert sim.num_levels == levels and sim.frameskip == 5
    ep = load_episode(h5, 1)
    all_levels = list(range(1, levels + 1))
    for mode in ('ground_truth', 'zero', 'random', 'reversed'):
        out = sim.dream(ep, all_levels, start_frame=0, context=3, horizon=6, action_mode=mode)
        for lv in all_levels:
            r = out[lv]
            assert len(r['pred_frames']) > 0, (lv, mode)
            assert r['pred_frames'].shape[1:] == (IMG, IMG, 3)
            assert len(r['latent_error']) == len(r['pred_frames'])
            assert r['decoded_with'] == ('decoder' if lv == 1 else 'nearest_neighbour')
    # Ground-truth actions must reproduce the teacher-forced prediction for the first step.
    enc = sim.encode(ep)
    hs = sim.history[0]
    first, _ = sim.rollout(enc, 1, 0, hs, 1)
    tf = sim.jepas[0].predict(enc['embeds'][1][:, :hs], enc['act_embs'][1][:, :hs])[:, -1]
    assert torch.allclose(first[0], tf[0], atol=1e-5)

    engine = InteractiveEngine(sim, h5, {})
    r = engine.reset({'episode': 2, 'start': 10})
    assert set(r['levels']) == {str(lv) for lv in all_levels}
    s = engine.step({'level': 1, 'mode': 'user', 'action': [0.3] * act_dim})
    assert s['frame'] and s['latent_error'] is not None
    engine.step({'level': 1, 'mode': 'zero'})
    engine.step({'level': 1, 'mode': 'random'})
    if levels > 1:
        roll = engine.rollout({'level': levels, 'mode': 'ground_truth', 'horizon': 3})
        assert len(roll['steps']) == 3
    with pytest.raises(ValueError):
        engine.step({'level': levels + 1})
