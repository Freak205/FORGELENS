# Blockers

## Active external access

AIForge-Doc v2 is free but Hugging Face marks it `gated: auto`. The local
adapter is ready and verified. The user must accept the dataset terms and
provide a read-only Hugging Face token through a secure mechanism before the
3.35 GB snapshot can be downloaded.

This does not block implementation work, but it blocks genuine dataset training
and evaluation.

Potential future blockers:

- AIForge-Doc v1 separately requires non-commercial-research acceptance and is
  not assumed authorized by v2 acceptance.
- Free Kaggle/Colab GPU execution may require user login or an API token.

Development proceeds with safe synthetic fixtures until either is actually
needed.
