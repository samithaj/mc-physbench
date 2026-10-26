# H-JEPA Studio

A web platform for **hierarchical JEPA world models**
([H-JEPA](https://github.com/kevinghst/H-JEPA), arXiv:2610.06805). Users can:

- sign in and **share world-model data** (episodes of frames + actions, in H-JEPA's HDF5 layout);
- **train** flat LeWM, H-JEPA (2–4 levels) or HWM models on **small rented RunPod GPUs**
  (RTX 5090 by default; RTX 4090/3090, A5000, L4, RTX 4000 Ada and A4000 also work);
- **evaluate** trained models with upstream's planning benchmark (50 start/goal tasks);
- **simulate** models in the browser: batch "dream" rollouts at every level of the
  hierarchy, or an interactive session where you step each level yourself and steer level 1
  with your own actions.

The 4 official datasets and 28 released checkpoints (seed 42, every env × variant) are
listed as public entries from the start, so users can simulate before training anything.

```
 browser ──HTTPS──▶  H-JEPA Studio (FastAPI + static UI, CPU only)
                      │  users, datasets, models, jobs, sessions (SQLite/Postgres)
                      │  blob storage on a volume, served through signed URLs
                      │  reaper: time limits, lost workers, idle sessions
                      ▼
                 RunPod REST API ── creates a pod from the worker image
                                          │
 GPU pod (worker/Dockerfile): upstream H-JEPA @ 97054ec + hjepa_worker agent
   job mode:   GET spec → download data/checkpoint → main_hjepa.py / eval.py / simulator
               → stream logs + Lightning metrics → upload artifacts → complete → pod deleted
   serve mode: load model, HTTP server on :8000 (only the platform holds its secret),
               the platform proxies the browser's step/rollout calls
```

## What a job runs

| UI action | On the GPU pod |
|---|---|
| Train | `python main_hjepa.py --config-name <env>_<variant> seed=… trainer.max_epochs=… loader.batch_size=…` (+ upstream's decoder training and planning eval callbacks) |
| Train on your data, built-in env | same config, with `data.dataset.name/val_name` and the probing datasets pointed at your files |
| Train on custom data | generates `config/train/custom_<variant>.yaml` from the Push-T configs with your image size, patch size, action dim and frameskip; probes off, decoders on, no planning eval |
| Planning eval | `python eval.py --config-name <env>_<planner> policy=<ckpt>` on the official eval tasks |
| Dream rollout | `hjepa_worker.simulator`: encode real frames at every level, roll each level's predictor forward under recorded or counterfactual actions (zero / random / reversed / scaled), decode with the trained `decoder_level{N}.ckpt` (or show the nearest real frame), report latent error per step |
| Interactive session | same simulator, kept in memory; step any level, choose level-1 actions with sliders, sample level-N macro-actions from the model's latent-action queue |

Official data is fetched on the pod exactly as the upstream README describes (Hugging Face
`jepa-world-models/h-jepa`, `quentinll/lewm-pusht`, `quentinll/lewm-cube`, including the
Push-T `block_ori` and Cube split preprocessing). Uploaded data is served to the pod through a
24-hour signed URL. When no validation set is given, the last 5% of episodes are held out.

## Data format

One HDF5 file per dataset, as read by `stable_worldmodel.data.HDF5Dataset`:

| column | shape | notes |
|---|---|---|
| `ep_len`, `ep_offset` | `(episodes,)` | episode `i` is rows `ep_offset[i] : ep_offset[i] + ep_len[i]` |
| `pixels` | `(N, H, H, 3)` uint8 | square frames, stored at training resolution |
| `action` | `(N, A)` float | one row per environment step |
| anything else | `(N, …)` | e.g. `proprio`, `state`; needed for the built-in env configs |

Uploads are validated on completion (structure, dtypes, episode ranges) and get a preview
strip. Built-in environments need their specific columns (shown per dataset); anything with
`pixels` + `action` can train as *Custom data*.

## Run it locally (no GPU needed)

```bash
cd hjepa_platform
pip install -e ".[test]"
HJS_LAUNCHER=local HJS_WORKER_MOCK=1 uvicorn hjepa_studio.app:app --port 8000
# open http://127.0.0.1:8000 and create an account (the first account is admin)
```

`HJS_WORKER_MOCK=1` makes workers fake the GPU work (synthetic losses, frames and
checkpoints), so the whole flow can be clicked through. With `HJS_LAUNCHER=local` and
without mock, workers run the real H-JEPA code as subprocesses; that needs a CUDA machine
with H-JEPA installed and `HJEPA_SRC` pointing to it.

Tests:

```bash
python -m pytest -q                         # platform + mock workers end to end
HJEPA_SRC=/path/to/H-JEPA python -m pytest  # also config generation against upstream YAML
```

`tests/test_simulator_contract.py` checks the simulator against a stand-in model with
upstream's `JEPA` interface (needs `torch`). `tests/test_simulator_real.py` builds real
H-JEPA models from upstream configs; it needs torch, hydra, transformers and the checkout.

## Deploy with RunPod

1. **Build the worker image** and push it somewhere RunPod can pull from: run the
   `hjepa-worker-image` GitHub workflow (pushes `ghcr.io/<owner>/hjepa-worker`, make the
   package public or add registry auth in RunPod), or
   `docker build -t <registry>/hjepa-worker hjepa_platform/worker`.
2. **Run the platform** on any small VM with a public HTTPS URL (pods call back into it):
   `cp .env.example .env`, fill in `HJS_SECRET_KEY`, `HJS_PUBLIC_URL`, `HJS_WORKER_IMAGE`
   and optionally a platform-paid `RUNPOD_API_KEY`, then `docker compose up -d` behind a TLS
   proxy (Caddy, nginx, Cloudflare Tunnel).
3. Users without a platform key add their own RunPod API key under **Settings**; it's stored
   encrypted and jobs then bill to their account.

Configuration (environment variables):

| variable | default | |
|---|---|---|
| `HJS_LAUNCHER` | `local` | `runpod` in production |
| `HJS_PUBLIC_URL` | `http://127.0.0.1:8000` | URL workers use to reach the platform |
| `HJS_SECRET_KEY` | generated into the data dir | set explicitly in production |
| `HJS_DATABASE_URL` | SQLite in `HJS_DATA_DIR` | e.g. `postgresql+psycopg://…` |
| `HJS_STORAGE_DIR` | `$HJS_DATA_DIR/storage` | put it on a persistent volume |
| `RUNPOD_API_KEY` | — | platform-paid GPUs |
| `HJS_WORKER_IMAGE` | `ghcr.io/samithaj/hjepa-worker:latest` | |
| `HJS_MAX_CONCURRENT_JOBS` | `3` | per user, jobs + sessions |
| `HJS_DEFAULT_MAX_HOURS` / `HJS_MAX_HOURS_LIMIT` | `6` / `48` | per-job time limit |
| `HJS_SESSION_IDLE_MINUTES` | `20` | idle sessions are stopped |
| `HJS_HEARTBEAT_TIMEOUT_S` | `900` | jobs whose worker goes silent are failed and their pod deleted |
| `HJS_CONTAINER_DISK_GB` | `60` | pod disk; raise it for DROID-scale data |

## Security model

- Passwords: scrypt. Sessions: 7-day HS256 JWTs. RunPod keys: Fernet-encrypted at rest.
- Each job/session gets its own random worker token (stored hashed); it only authorizes
  callbacks for that job/session.
- Files move only through HMAC-signed, expiring URLs; workers can upload only under
  `jobs/<id>/`. Storage keys are validated against path traversal.
- Private datasets/models are invisible to other users (404, not 403).
- Hydra overrides from users are passed as argv (never through a shell) and restricted to
  plain `key=value` with no `hydra.*`, `subdir` or output-path keys.
- Interactive simulators are never exposed directly: the platform proxies a fixed set of
  calls and authenticates to the worker with a per-session secret.

## Known gaps

- Not yet run on a real RunPod GPU from this repo: the mock end-to-end tests cover the
  platform ↔ worker protocol, the simulator is tested against a stand-in for upstream's model
  interface, and config generation is tested against upstream's YAML. Run one short training
  job (for example FourRoom with `hjepa_l2`, 1 epoch) after deploying.
- GPU type ids assume RunPod's naming (`NVIDIA GeForce RTX 5090`, …); check them in the
  RunPod console if pod creation reports an unknown GPU type.
- Whether the released Hugging Face checkpoints ship pixel decoders is not confirmed. Without
  them, simulations show the nearest real frame for each predicted latent.
- Upper levels with proprio fusion are not supported by the simulator; none of the upstream
  configs use them.
- Storage is a local volume (no S3 backend yet). The schema is created with `create_all`
  (no migrations yet).
- DROID is not offered: its training set is 91 GB, too large for the small-GPU pods.
