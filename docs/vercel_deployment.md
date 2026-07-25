# Vercel deployment audit

## Bundle root cause

The original Vercel function installed every base dependency from
`pyproject.toml`. A clean Python 3.14 manylinux installation measured
817.76 MiB before application files:

| Installed content | Uncompressed size |
| --- | ---: |
| `torch` | 666.97 MiB |
| NumPy (`numpy` and `numpy.libs`) | 55.03 MiB |
| SymPy | 25.37 MiB |
| Pillow (`PIL` and `pillow.libs`) | 18.65 MiB |
| Other Python dependencies | 51.74 MiB |
| **Dependency total** | **817.76 MiB** |

The tracked source, manifests, and 0.40 MiB checkpoint brought the Vercel
trace to the reported 820.66 MB. Repository exclusions could remove only the
small remainder; PyTorch itself made the standard 500 MB function impossible.

## Deployment architecture

The hosted `POST /predict` function now loads `api/model.onnx`, an exported and
numerically verified representation of the same
`ResidualUNetJointDetector` checkpoint. The response fields, thresholds,
temperature scaling, mask encoding, safety warnings, upload limit, and
`/predict` rewrite are unchanged.

Vercel installs only:

- `onnxruntime==1.27.0`
- `numpy==2.3.2`
- `Pillow==12.3.0`
- their small transitive dependencies

PyTorch, torchvision, Hugging Face Hub, Pydantic, PyYAML, and structlog are in
the optional `research` dependency set. `tools/run.ps1 sync` installs both the
development and research extras for local training and evaluation.

## Final measured function content

A clean Python 3.14 `x86_64-manylinux_2_28` installation from the locked base
dependencies was measured uncompressed:

| Installed content | Uncompressed size |
| --- | ---: |
| ONNX Runtime | 50.54 MiB |
| NumPy (`numpy` and `numpy.libs`) | 55.03 MiB |
| Pillow (`PIL` and `pillow.libs`) | 18.65 MiB |
| Protobuf, packaging, FlatBuffers, metadata | 2.18 MiB |
| API entrypoint and ONNX model | 0.19 MiB |
| **Measured function content** | **126.59 MiB** |
| **Headroom below 500 MiB** | **373.41 MiB** |

The measurement uses the same platform wheel tags Vercel resolves for Python
3.14. It deliberately reports uncompressed installed content, which is the
relevant function-size measure.

## Exclusions

`vercel.json` excludes all research source, tests, datasets, configuration
manifests, reports, documentation, experiment results, checkpoints, caches,
notebooks, tooling, and repository metadata from the Python function.
`includeFiles` contains only `api/model.onnx`.

`.vercelignore` additionally prevents those assets from entering Vercel CLI
uploads. The build retains `pyproject.toml`, `uv.lock`, and `src/` only long
enough to resolve and install the project; `src/` and build metadata are
excluded from the packaged function.

## Reproducing the model

After installing the local research environment, regenerate and verify the
deployment artifact with:

```powershell
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe scripts\export_vercel_model.py
```

The exporter checks the ONNX graph and compares both output tensors against
PyTorch before succeeding.
