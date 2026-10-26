"""Config generation and command building. Reads upstream YAML configs from a copy of
an H-JEPA checkout (set HJEPA_SRC); nothing from the checkout is executed."""

import importlib
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

yaml = pytest.importorskip('yaml')
SRC = Path(os.environ.get('HJEPA_SRC', '/opt/H-JEPA'))


@pytest.fixture
def recipes(tmp_path, monkeypatch):
    if not (SRC / 'h_jepa/config/train/base/pusht.yaml').exists():
        pytest.skip('H-JEPA checkout not found (set HJEPA_SRC)')
    shutil.copytree(SRC / 'h_jepa/config', tmp_path / 'src/h_jepa/config')
    monkeypatch.setenv('HJEPA_SRC', str(tmp_path / 'src'))
    monkeypatch.setenv('HJEPA_HOME', str(tmp_path / 'home'))
    import hjepa_worker.recipes as mod
    return importlib.reload(mod)


@pytest.mark.parametrize('variant,levels', [('lewm', 1), ('hjepa_l3', 3), ('hwm_l2', 2)])
def test_custom_configs(recipes, variant, levels):
    name = recipes.write_custom_configs(variant, 'ds_a', 'ds_a_val', img_size=96, action_dim=7,
                                        frameskip=3, levels=levels)
    assert name == f'custom_{variant}'
    cfg_dir = recipes.H_JEPA_DIR / 'config'
    base_text = (cfg_dir / 'train/base/custom.yaml').read_text()
    assert base_text.startswith('# @package _global_')
    base = yaml.safe_load(base_text)
    assert base['data']['dataset']['name'] == 'ds_a' and base['data']['dataset']['val_name'] == 'ds_a_val'
    assert base['data']['dataset']['keys_to_load'] == ['pixels', 'action']
    assert base['data']['dataset']['level1']['frameskip'] == 3
    assert base['img_size'] == 96 and 96 % base['patch_size'] == 0
    assert base['level1']['wm']['action_dim'] == 7
    assert base['planning_eval']['enabled'] is False
    model = yaml.safe_load((cfg_dir / f'train/{name}.yaml').read_text())
    assert model['defaults'][0] == 'base/custom'
    assert model['output_model_name'] == name
    probes = model['final_probing_decoding_eval']['overrides']
    assert all(probes[f'level{lv}']['probes']['enabled'] is False for lv in range(1, levels + 1))
    if levels > 1:
        assert f'level{levels}' in model  # upper-level architecture carried over from Push-T
    probing = yaml.safe_load((cfg_dir / 'probing/custom.yaml').read_text())
    assert probing['train_dataset']['name'] == 'ds_a' and probing['eval_dataset']['name'] == 'ds_a_val'


def test_patch_size(recipes):
    assert recipes._patch_size_for(224) == 14
    assert recipes._patch_size_for(64) == 4
    with pytest.raises(ValueError):
        recipes._patch_size_for(97)


def test_train_command(recipes):
    p = {'preset': 'ant', 'variant': 'hjepa_l3', 'seed': 7, 'epochs': 2, 'batch_size': 64,
         'run_planning_eval': True, 'train_decoders': True, 'extra_overrides': ['level2.lr=2e-5']}
    official = recipes.train_command(p, 'ant_hjepa_l3', 'tr', 'va', official=True, levels=3)
    assert '--config-name=ant_hjepa_l3' in official and 'seed=7' in official
    assert 'planning_eval.enabled=true' in official and official[-1] == 'level2.lr=2e-5'
    assert not any(a.startswith('data.dataset.name') for a in official)
    user = recipes.train_command(p, 'ant_hjepa_l3', 'ds_x', 'ds_y', official=False, levels=3)
    assert 'data.dataset.name=ds_x' in user and 'data.dataset.val_name=ds_y' in user
    assert '++final_probing_decoding_eval.overrides.eval_dataset.name=ds_y' in user
    assert recipes.run_dir_for(p, 'ant_hjepa_l3').parts[-3:] == ('ant', 'ant_hjepa_l3', 'seed7')


def test_split_h5(recipes, tmp_path):
    src = tmp_path / 'src.h5'
    lens = np.array([10, 12, 8, 9, 11] * 4)
    with h5py.File(src, 'w') as f:
        f['ep_len'] = lens
        f['ep_offset'] = np.concatenate([[0], np.cumsum(lens)[:-1]])
        f['pixels'] = np.arange(lens.sum())[:, None, None, None].repeat(4, 1).repeat(4, 2).repeat(3, 3).astype(np.uint8)
        f['action'] = np.arange(lens.sum(), dtype=np.float32)[:, None]
    n_tr, n_va = recipes.split_h5(src, tmp_path / 'tr.h5', tmp_path / 'va.h5')
    assert (n_tr, n_va) == (19, 1)
    with h5py.File(tmp_path / 'va.h5') as f:
        assert list(f['ep_len'][:]) == [11] and f['ep_offset'][0] == 0
        assert f['action'][0, 0] == lens[:-1].sum()
    with h5py.File(tmp_path / 'tr.h5') as f:
        assert f['pixels'].shape[0] == lens[:-1].sum()
