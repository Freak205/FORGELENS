# Published Baseline Reproduction

## TruFor

ForgeLens compares against the official TruFor implementation for image-level
forgery detection and pixel-level localization. The source checkout is pinned
at commit `ae54475df6f41a491d7615100feb19263dec13f7` in
`F:\HYPERVERGE\third_party\TruFor`. Its licence permits free informational and
nonprofit use, requires attribution and preservation of notices, and prohibits
industrial or profit-oriented use. ForgeLens does not redistribute its code or
weights.

The exact source, licence checksums, official weights URL, and official MD5 are
recorded in `configs/baselines/third_party.lock.json`. On 2026-07-23 the
official GRIP server timed out from this machine, so no unverified mirror is
used and no TruFor result is claimed yet.

TruFor's official environment uses PyTorch 1.11/CUDA 11.3 and old package
versions. Reproduction runs in the upstream isolated Docker environment, not in
ForgeLens's Python environment. This prevents dependency drift and isolates the
upstream checkpoint load, which uses Python pickle semantics.

Prepare the immutable CORD test input index with:

```powershell
.\tools\run.ps1 prepare-trufor
```

Once the official weights are available and their MD5 is verified, execute the
upstream container against the indexed images, normalize each `.npz` through
`forgelens.baselines.trufor.load_trufor_output`, and evaluate it with the same
locked split, metrics, bootstrap seed, and calibration protocol as ForgeLens.
Localization metrics are computed on forged examples only.
