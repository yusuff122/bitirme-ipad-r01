# Experiments

| ID | Status | Description |
|---|---|---|
| E0 | Complete | ResNet18 frame embeddings and normal memory bank; validation threshold; frame metrics. |
| E1 | Complete | Frozen 16-frame/stride-4 Video Swin Tiny features and k-NN; diagnostic analysis found weak test performance, with no mapping bug. |
| E2 | Implemented; smoke passed | GRU next-embedding predictor over frozen E1 features; full training intentionally not started. |
| E3 | Deferred | Periodic memory, sliding window, reconstruction, LoRA. |

Record config, split manifest, threshold protocol, software/device and metrics with every run. Do not compare runs with different test splits without stating this.
