# ForgeLens Model Card

Status: research prototype. No operational performance is claimed.

The current `TinyUNetJointDetector` checkpoint was trained on a 1,966-sample
paired GPT-Image-2/CORD benchmark. Test ROC-AUC was 0.502 (95% CI
0.444–0.558), false-positive rate was 1.0 at the validation-selected threshold,
and pixel IoU was 0.020. It is rejected and must not be used for decisions.

Intended future use is controlled research on licensed receipt/form datasets.
Real identity documents, biometrics, forensic conclusions, autonomous rejection,
and production KYC decisions are out of scope.
