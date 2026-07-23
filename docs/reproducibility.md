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
.\tools\run.ps1 build-cord-copy-move
.\tools\run.ps1 train-real-baseline
```

Research experiments must add an immutable split manifest, resolved
configuration, seed, Git commit, hardware record, checkpoint hash, and metrics
record. `SMOKE-0001` verifies plumbing only and is not a scientific result.
Published baselines are revision- and licence-pinned in
`configs/baselines/third_party.lock.json`.

## Clean-install audit

Commit `605bb90` was built as a wheel and installed into a new isolated
environment at `F:\HYPERVERGE\.tmp\clean-audit-605bb90`. An isolated package
import, residual-model construction, and forward-shape smoke passed using the
CPU PyTorch wheel. The wheel checksum and audit evidence are recorded in
`reports/tables/clean_install_audit.json`.

## Credentials

Accept gated dataset terms in the browser, then run
`.\tools\setup-secrets.ps1`. Credentials are encrypted with Windows DPAPI for
the current Windows user and stored only below `F:\HYPERVERGE\secrets`.
Plaintext secrets are never committed or logged.
