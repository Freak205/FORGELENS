# What Three Failed Detectors Taught Me About Document Forgery Research

ForgeLens began with the smallest defensible model, not a dashboard or a large
VLM. I built strict provenance, source-group-safe splits, joint image/mask
training, calibration, and bootstrap evaluation before increasing complexity.

The first RGB model reached test ROC-AUC 0.548, but its 95% interval
(0.468–0.628) included chance and its selected operating point falsely flagged
99% of authentic documents. A skip-connected localizer and a higher-resolution
residual model did not fix the problem. I rejected all three.

The important result is methodological: copy-move derivatives are useful for
testing plumbing, but not a credible substitute for AI-inpainted receipt edits.
The next valid experiment must use licensed generator-shift data. ForgeLens
therefore treats uncertainty and negative results as first-class artifacts,
and its demo defaults to manual review.
