# Model files

The gateway loads one file per `local` model from `$MODELS_DIR` (`/models`
inside the container):

- `<model-name>.pt` — torch state dict for the tiny `SignalGRU`
  (`{"state_dict": ..., "window": 30}`), used when the `gpu` extra is installed.
- `<model-name>.json` — optional `{"gain": 1.0}` scale params for the NumPy
  fallback. The gateway works without any files (untrained baseline).

Train a dummy model end to end (works with or without torch):

```bash
python -m inference.scripts.train_dummy --out /models/local-gru.pt
```

With torch installed this writes real GRU weights; without torch it writes a
`local-gru.json` sidecar. Weights are runtime artifacts — never commit large
`.pt` files; keep a small checked-in dummy under `inference/models/` only if
needed for CI smoke tests.
