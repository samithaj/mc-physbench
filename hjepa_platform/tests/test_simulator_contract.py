"""Simulator logic against a stand-in model with upstream's JEPA interface
(h_jepa/models/jepa.py: encode / predict / action_embed / action_encoder,
temporal_stride / temporal_window_size, chunk_temporal_inputs). Checks shapes,
level alignment and action handling without needing the H-JEPA checkout."""

import numpy as np
import pytest

torch = pytest.importorskip('torch')
nn = torch.nn

IMG, D, A_RAW, FS = 16, 12, 2, 5


class FakeJEPA(nn.Module):
    def __init__(self, level, stride=2, window=1, a_in=A_RAW * FS):
        super().__init__()
        self.level = level
        self.temporal_stride = stride
        self.temporal_window_size = window
        in_dim = 3 * IMG * IMG if level == 1 else D
        self.encoder = nn.Linear(in_dim, D)
        self.latent_dim = 4
        self.action_encoder = nn.Identity() if level == 1 else nn.Linear(a_in, self.latent_dim)
        self.action_embed = nn.Linear(a_in if level == 1 else self.latent_dim, 8)
        self.predictor = nn.Linear(D + 8, D)

    def encode(self, info, key='pixels', temporal_stride=None, chunk_temporal_inputs=False):
        x = info[key]
        act = info.get('action')
        if chunk_temporal_inputs:
            s = temporal_stride or self.temporal_stride
            w = self.temporal_window_size
            n = (x.shape[1] - w) // s + 1
            idx = [j * s + w - 1 for j in range(n)]
            x = x[:, idx]
            if act is not None:
                act = torch.stack([act[:, i:i + s].mean(1) for i in idx], dim=1)
        b, t = x.shape[:2]
        info['embed_0'] = self.encoder(x.reshape(b, t, -1).float())
        if act is not None:
            info['action_0'] = self.action_embed(self.action_encoder(act))
        return info

    def predict(self, emb, act_emb):
        return emb + 0.1 * torch.tanh(self.predictor(torch.cat([emb, act_emb], -1)))

    def get_latent_action_queue(self):
        return None if self.level == 1 else torch.randn(10, self.latent_dim)


class FakeHJEPA(nn.Module):
    def __init__(self, levels):
        super().__init__()
        self.jepas = nn.ModuleList([FakeJEPA(1)] + [FakeJEPA(lv, a_in=8) for lv in range(2, levels + 1)])


def _files(tmp_path, levels):
    from omegaconf import OmegaConf

    model = FakeHJEPA(levels)
    torch.save(model, tmp_path / 'm_object.ckpt')
    torch.save({'format': 'lejepa_training_normalizer_v1',
                'stats': {'action': {'mean': np.zeros(A_RAW, np.float32), 'std': np.ones(A_RAW, np.float32)}}},
               tmp_path / 'normalizer.pt')
    cfg = {'data': {'dataset': {'level1': {'frameskip': FS}}}, 'img_size': IMG}
    for lv in range(1, levels + 1):
        cfg[f'level{lv}'] = {'wm': {'history_size': 3}}
    OmegaConf.save(OmegaConf.create(cfg), tmp_path / 'config.yaml')
    import h5py

    eps, length = 3, 100
    with h5py.File(tmp_path / 'data.h5', 'w') as f:
        f['ep_len'] = np.full(eps, length, np.int64)
        f['ep_offset'] = np.arange(eps, dtype=np.int64) * length
        f['pixels'] = np.random.randint(0, 255, (eps * length, IMG, IMG, 3), np.uint8)
        f['action'] = np.random.randn(eps * length, A_RAW).astype(np.float32)


@pytest.mark.parametrize('levels', [1, 2, 3])
def test_dream_alignment(tmp_path, levels):
    from hjepa_worker.simulator import WorldModelSimulator, load_episode

    _files(tmp_path, levels)
    sim = WorldModelSimulator(tmp_path / 'm_object.ckpt', tmp_path / 'normalizer.pt',
                              tmp_path / 'config.yaml', {}, device='cpu')
    ep = load_episode(tmp_path / 'data.h5', 1)
    enc = sim.encode(ep)
    assert enc['n1'] == 20  # 100 raw steps / frameskip 5
    for lv in range(1, levels + 1):
        n = enc['embeds'][lv].shape[1]
        assert n == (20 if lv == 1 else (enc['embeds'][lv - 1].shape[1] - 1) // 2 + 1)
        assert enc['act_embs'][lv].shape[1] == n
        assert len(enc['frame_index'][lv]) == n
        assert enc['frame_index'][lv][1] - enc['frame_index'][lv][0] == FS * 2 ** (lv - 1)

    # First predicted step equals the teacher-forced prediction from the same context.
    first, ctx = sim.rollout(enc, 1, 2, 3, 4)
    assert ctx == 3 and first.shape == (4, D)
    tf = sim.jepas[0].predict(enc['embeds'][1][:, 2:5], enc['act_embs'][1][:, 2:5])[:, -1]
    assert torch.allclose(first[0], tf[0], atol=1e-6)

    out = sim.dream(ep, list(range(1, levels + 1)), start_frame=10, context=2, horizon=50)
    for lv, r in out.items():
        n = enc['embeds'][lv].shape[1]
        assert 0 < len(r['pred_frames']) <= n
        assert len(r['latent_error']) <= len(r['pred_frames'])
        assert r['decoded_with'] == 'nearest_neighbour'
        assert r['pred_frames'].shape[1:] == (IMG, IMG, 3)

    zero = sim.dream(ep, [1], 10, 2, 5, action_mode='zero')
    gt = sim.dream(ep, [1], 10, 2, 5)
    assert zero[1]['latent_error'] != gt[1]['latent_error']  # actions matter
    with pytest.raises(ValueError):
        sim.dream(ep, [1], 0, 2, 5, action_mode='teleport')


def test_interactive_engine(tmp_path):
    from hjepa_worker.engine import InteractiveEngine
    from hjepa_worker.simulator import WorldModelSimulator

    _files(tmp_path, 2)
    sim = WorldModelSimulator(tmp_path / 'm_object.ckpt', tmp_path / 'normalizer.pt',
                              tmp_path / 'config.yaml', {}, device='cpu')
    engine = InteractiveEngine(sim, tmp_path / 'data.h5', {})
    with pytest.raises(ValueError):
        engine.step({'level': 1})
    r = engine.reset({'episode': 2, 'start': 20})
    assert r['episode'] == 2 and set(r['levels']) == {'1', '2'}
    assert engine.info()['action_dim'] == A_RAW
    s = engine.step({'level': 1, 'mode': 'user', 'action': [0.5, -0.5]})
    assert s['frame'] and s['latent_error'] is not None and s['raw_step'] == 25
    for mode in ('zero', 'random', 'ground_truth'):
        engine.step({'level': 1, 'mode': mode})
    s2 = engine.step({'level': 2, 'mode': 'random'})  # sampled from the latent-action queue
    assert s2['raw_step'] == 30
    roll = engine.rollout({'level': 2, 'horizon': 20})
    assert len(roll['steps']) == 20
    assert roll['steps'][-1]['real'] is None  # ran past the end of the episode
    with pytest.raises(ValueError):
        engine.step({'level': 2, 'mode': 'user', 'action': [0, 0]})
    assert len(engine.sample_action({})['action']) == A_RAW
