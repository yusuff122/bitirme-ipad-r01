# Dataset notes

Target scenario: `R01_conveyer_belt_real` from IPAD. The extracted data is at `/Users/yusuf/Downloads/IPAD_dataset/R01`; `data/r01` is a symlink to that folder, so dataset files are not duplicated. No data were downloaded or copied. The archive uses the short directory name `R01`.

## Inspected structure

- `training/frames/01` through `34`: 34 sequences, 7,808 JPG frames total.
- `testing/frames/01` through `15`: 15 sequences, 3,685 JPG frames total.
- `test_label/001.npy` through `015.npy`: one 1D label vector per test sequence; values are 0 (normal) and 1 (anomaly). Each vector length matches its sequence frame count.
- Test labels total 3,685: 2,431 normal and 1,254 anomaly frames.
- Sample frame resolution: 256 x 256 pixels. Frame names are numeric JPGs; sort numerically.
- `test_label/001.npy` maps to `testing/frames/01` (zero-padded label basename maps to sequence directory).

Preserve sequence boundaries and split training sequences into train/validation. Never choose a threshold on test data.
