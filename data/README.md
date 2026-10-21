# Yerel veri

IPAD arşivini veya kopyasını Git deposuna eklemeyin. `data/r01/` yok sayılır. ZIP'ten yalnızca `R01_conveyer_belt_real/` çıkarmak için:

```bash
python scripts/extract_r01.py --zip /mutlak/yol/ipad_dataset.zip
```

Arşiv açılmışsa doğrudan kaynak dizini kullanarak inceleyin:

```bash
python scripts/inspect_dataset.py --root /mutlak/yol/R01_conveyer_belt_real
```
