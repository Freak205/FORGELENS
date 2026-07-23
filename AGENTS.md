# ForgeLens Agent Rules

## Mission

Build a reproducible, research-grade prototype for calibrated and explainable
multimodal document forgery detection under distribution shift. Never describe
the system as production-ready or as forensic proof.

## Safety and integrity

- Use only fictional, non-sensitive documents and appropriately licensed data.
- Never collect biometrics or use real identity documents.
- Do not build or publish an operational forgery generator.
- Never fabricate metrics, citations, experiments, or dataset access.
- Fit models, thresholds, and calibration only on training/validation data.
- Keep source documents, templates, identities, and generators disjoint across
  splits where applicable.
- Preserve an `uncertain` verdict and a `manual_review` action.
- Do not commit secrets, gated data, downloaded datasets, or large checkpoints.
- Keep every ForgeLens artifact, environment, cache, dataset, and output under
  `F:\HYPERVERGE`.

## Engineering rules

- Python 3.14, PyTorch, type hints, structured logging, validated config.
- Set and record random seeds; default deterministic evaluation.
- Store exact resolved config beside checkpoints and results.
- Every experiment records hypothesis, data version, split manifest, seed,
  commit, hardware, duration, peak VRAM, metrics, artifacts, and decision.
- Add architecture complexity only after measured baseline failure analysis.
- Prefer free/local tools and free GPU routes. Never incur cost without approval.

## Required verification

Run after meaningful changes:

```powershell
.\tools\run.ps1 format-check
.\tools\run.ps1 lint
.\tools\run.ps1 typecheck
.\tools\run.ps1 test
```

Training and evaluation changes also require a CPU smoke train, inference
integration test, and deterministic evaluation rerun.

## Persistent handoff

Continuously update `PROJECT_STATE.md`, `TASKS.md`, `DECISIONS.md`,
`BLOCKERS.md`, and `docs/experiment_log.md`. A new session must be able to
resume from `PROJECT_STATE.md` without reconstructing context.

## Definition of done

Completion requires an installable repository, passing verification, documented
and legally usable datasets, leakage-safe splits, genuinely trained
classification and localization models, an attributed published baseline or
evidence-supported reproduction blocker, calibration and abstention analysis,
cross-domain/generalization and robustness evaluation, VLM zero-shot versus
LoRA/QLoRA comparison, traceable checkpoints, reproducible numbers, measured
latency/VRAM, paper-style report, working safe demo, and a final audit finding
no unsupported claims.
