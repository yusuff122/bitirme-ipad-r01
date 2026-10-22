# Project plan

1. Inspect the exact R01 layout and annotation semantics before defining splits or labels.
2. Prepare a sequence-aware manifest. Keep whole sequences in one partition.
3. E0: freeze ImageNet ResNet18, build a normal-train embedding bank, select threshold on validation, evaluate once on test.
4. E1: generate 16-frame clips at stride 4 within each sequence, extract frozen Video Swin Tiny features, train normal clip bank, map clip scores back to frame timeline.
5. E2: train a GRU next-embedding predictor on frozen E1 features; use only validation normal prediction loss for early stopping and a validation frame-level threshold.
6. Consider E3 only after the requested baselines and research comparison are complete.

No experiment metrics are claimed until a run produces them.
