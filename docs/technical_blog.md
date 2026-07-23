# What Four Failed Detectors Taught Me About Document Forgery Research

ForgeLens began with the smallest defensible model, not a dashboard or a large
VLM. I built strict provenance, source-group-safe splits, joint image/mask
training, calibration, and bootstrap evaluation before increasing complexity.

The first RGB model reached test ROC-AUC 0.548, but its 95% interval
(0.468–0.628) included chance and its selected operating point falsely flagged
99% of authentic documents. A skip-connected localizer and a higher-resolution
residual model did not fix the problem. I rejected all three.

The important result is methodological: copy-move derivatives are useful for
testing plumbing, but not a credible substitute for AI-inpainted receipt edits.
The licensed paired GPT-Image-2 experiment then reached only ROC-AUC 0.502
(95% CI 0.444–0.558) and pixel IoU 0.020. That ruled out a tempting story that
real inpainting data alone would rescue the tiny RGB architecture. ForgeLens
therefore treats uncertainty and negative results as first-class artifacts,
and its demo defaults to manual review.
