# Reproducibility

## Storage boundary

Clone or copy the repository to `F:\HYPERVERGE\forgelens`. The task runner
forces uv, temporary, Hugging Face, Torch, and XDG caches below
`F:\HYPERVERGE`.

## Verified commands

```powershell
.\tools\run.ps1 sync
.\tools\run.ps1 cuda
.\tools\run.ps1 verify
.\tools\run.ps1 smoke-train
.\tools\run.ps1 prepare-trufor
```

Research experiments must add an immutable split manifest, resolved
configuration, seed, Git commit, hardware record, checkpoint hash, and metrics
record. `SMOKE-0001` verifies plumbing only and is not a scientific result.
Published baselines are revision- and licence-pinned in
`configs/baselines/third_party.lock.json`.
