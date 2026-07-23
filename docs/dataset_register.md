# Dataset and Model Asset Register

Status values: `candidate`, `verified-usable`, `gated`, `rejected`.

No candidate is approved until its official source, paper, licence text, access
terms, intended-use compatibility, version/checksum, contents, privacy risks,
split strategy, and leakage risks are verified.

| Asset | Status | Official source | Licence/access | Decision |
|---|---|---|---|---|
| AIForge-Doc v1 | candidate | [paper](https://arxiv.org/abs/2602.20569), [dataset card](https://huggingface.co/datasets/Scam-AI/AIForge-Doc-v1) | derived from CORD, WildReceipt, SROIE, XFUND; source-specific terms require per-subset enforcement | promising primary GenAI benchmark; do not download until provenance manifest is audited |
| AIForge-Doc v2 | gated | [dataset card](https://huggingface.co/datasets/Scam-AI/AIForge-Doc-v2) | Hugging Face API reports `gated: auto`; card body says CC BY 4.0, front matter says CC BY-NC-SA 4.0, gate says academic/non-commercial, and source terms vary | treat as non-commercial research only; user acceptance and read token required; filter by source licence |
| FantasyID | candidate | [official page](https://www.idiap.ch/paper/fantasyid/) | official page states public commercial/non-commercial availability; exact archive licence file still required | strong safe-design shift set; contains permissively sourced real faces, so privacy/biometric review is required before use |
| DocTamper | gated | [official repository](https://github.com/qcf-568/DocTamper) | non-commercial; university/research-institute application and signed form required | cannot be used locally without eligibility and approval; code/weights licence also requires file-level audit |
| T-SROIE | candidate | [upstream repository](https://github.com/wangyuxin87/Tampered_sroie) | exact dataset licence not yet located; SROIE carries research-use terms | do not use until upstream licence and redistribution chain are verified |
| ForensicHub | candidate | [documentation](https://scu-zjz.github.io/ForensicHub-doc/) | PyPI metadata says educational use/CC BY 4.0; repository file-level licences and model-weight terms remain to audit | evaluate as an adapter/reproduction route, not as blanket permission for bundled assets |

## Verified dataset characteristics

### AIForge-Doc v1

- 4,061 forged receipt/form images from CORD, WildReceipt, SROIE, and XFUND.
- Nine languages; numeric-field edits by Gemini 2.5 Flash Image and Ideogram v2
  Edit; pixel masks in DocTamper-compatible format.
- Required split rule: group by original source image/template and hold generator
  out explicitly; never allow paired variants across splits.
- Privacy risk: inherited source-document content must be inspected and handled
  according to each source corpus.

### AIForge-Doc v2

- 3,066 GPT-Image-2 forgeries; 3,062 have same-spec v1 pairs according to the
  dataset card.
- Required split rule: paired v1/v2 samples and their pristine source belong to
  the same partition. Use generator as an explicit distribution-shift axis.

### FantasyID

- 362 designed cards from 13 templates and ten language families are reported
  by the DeepID paper; 786 bona fide captures and 1,572 forged samples were used
  for train/validation in the challenge.
- Text is fictional, but faces are of real people from public face datasets.
  ForgeLens will not download or process it until a documented biometric/privacy
  exception is justified; safer receipt/form datasets take priority.

### DocTamper

- Traditional text replacement/insertion/removal with pixel localization.
- Gated to eligible academic applicants and non-commercial research. This is
  not an active user blocker because development continues with safe fixtures
  and other candidates.

## Selection direction

Primary route: safe fictional fixtures for engineering, followed by a
licence-filtered AIForge-Doc subset for GenAI distribution shift. DocTamper is a
published reference/reproduction target but its gated dataset is not assumed
available. FantasyID is deferred because its real-person faces conflict with the
project's conservative no-biometric posture.

## Authentic source corpus

CORD v2 is selected as the first authentic receipt source. Official metadata
identifies a CC BY 4.0 licence, 800 train, 100 validation, and 100 test images.
ForgeLens pins revision
`7f0115a4b758a71d6473b8d085751692da2fef98` and preserves official splits.
Its accession record is stored beside local data; attribution must appear in
the final report and README.

## Derived traditional-tampering benchmark

`forgelens/cord-copy-move-v1` contains one deterministic copy-move derivative
and exact binary target mask for each of the 1,000 pinned CORD images. It also
indexes the authentic source, producing 2,000 balanced samples while retaining
CORD's official 1,600/200/200 train/validation/test sample counts. Each
authentic/forged pair shares a source group and cannot cross splits. This is a
traditional-tampering engineering benchmark, not a substitute for AIForge's
AI-inpainting evaluation. Generated assets remain below
`F:\HYPERVERGE\data\cord-copy-move-v1`; checksums and provenance are locked in
`configs/data/cord_copy_move_v1.json`.
