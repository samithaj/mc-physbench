"""Materialize job inputs (datasets, model checkpoints) under HJEPA_HOME on the worker."""

from __future__ import annotations

import shutil

from .client import download
from .recipes import HJEPA_HOME, fetch_official_data, split_h5


def fetch_dataset(item: dict, log=print) -> dict:
    """Return {'train': name, 'val': name|None, 'path': Path} with files in HJEPA_HOME."""
    HJEPA_HOME.mkdir(parents=True, exist_ok=True)
    source = item['source']
    if source == 'official':
        log(f'fetching official {item["preset"]} data (Hugging Face)')
        train, val = fetch_official_data(item['preset'], log=log)
        return {'train': train, 'val': val, 'path': HJEPA_HOME / f'{train}.h5',
                'val_path': HJEPA_HOME / f'{val}.h5'}
    name = item['name']
    dest = HJEPA_HOME / f'{name}.h5'
    if not dest.exists():
        if source == 'url':
            log(f'downloading {name} ({item.get("size_bytes", 0) / 1e9:.2f} GB) from the platform')
            download(item['url'], dest)
        elif source == 'huggingface':
            from huggingface_hub import hf_hub_download

            log(f'downloading {item["repo"]}/{item["filename"]} from Hugging Face')
            path = hf_hub_download(repo_id=item['repo'], filename=item['filename'],
                                   repo_type=item.get('repo_type', 'dataset'),
                                   revision=item.get('revision'), local_dir=str(HJEPA_HOME / 'hf'))
            shutil.move(path, dest)
        else:
            raise ValueError(f'unknown dataset source {source!r}')
    return {'train': name, 'val': None, 'path': dest}


def ensure_val_split(train: dict, log=print) -> tuple[str, str]:
    """Hold out the last 5% of episodes when no validation set was given."""
    name = train['train']
    tr, va = f'{name}_trainsplit', f'{name}_valsplit'
    if not (HJEPA_HOME / f'{va}.h5').exists():
        n_tr, n_va = split_h5(train['path'], HJEPA_HOME / f'{tr}.h5', HJEPA_HOME / f'{va}.h5')
        log(f'no validation set given: held out {n_va} of {n_tr + n_va} episodes')
    return tr, va


def fetch_model(item: dict, log=print) -> dict:
    """Download a model; return paths {ckpt, normalizer, config, decoders: {level: path}}."""
    if item['source'] == 'huggingface':
        from huggingface_hub import snapshot_download

        root = HJEPA_HOME / 'ckpts_hf'
        prefix = item['prefix']
        log(f'downloading checkpoint {item["repo"]}/{prefix} from Hugging Face')
        snapshot_download(repo_id=item['repo'], allow_patterns=[f'{prefix}/*'],
                          ignore_patterns=['*/spt/*', '*/lightning_resume/*', '*_epoch_*'],
                          local_dir=str(root))
        mdir = root / prefix
        ckpt = mdir / item['object_ckpt']
        if not ckpt.exists():
            found = sorted(mdir.glob('*_object.ckpt'))
            if not found:
                raise FileNotFoundError(f'no *_object.ckpt under {prefix}')
            ckpt = found[0]
    else:
        mdir = HJEPA_HOME / 'ckpts' / 'platform' / 'model'
        files = item['files']
        names = {'object_ckpt': 'model_object.ckpt', 'normalizer': 'normalizer.pt',
                 'config': 'config.yaml'}
        for logical, url in files.items():
            if logical.startswith('decoder_level'):
                dest = mdir / 'final_probing_decoding_eval' / f'{logical}.ckpt'
            elif logical in names:
                dest = mdir / names[logical]
            else:
                continue
            log(f'downloading {logical}')
            download(url, dest)
        ckpt = mdir / 'model_object.ckpt'
    decoders = {}
    for path in (mdir / 'final_probing_decoding_eval').glob('decoder_level*.ckpt'):
        decoders[int(path.stem.removeprefix('decoder_level'))] = path
    return {'dir': mdir, 'ckpt': ckpt, 'normalizer': mdir / 'normalizer.pt',
            'config': mdir / 'config.yaml', 'decoders': decoders}
