# Industrial Conveyor Video Anomaly Detection using IPAD R01

Gerçek endüstriyel konveyör kamera videolarında normal ve anormal davranışları tespit etmek için video anomali tespiti altyapısı. Bu depo IPAD veri setinin `R01_conveyer_belt_real` senaryosunu hedefler. Veri seti bu klasöre kopyalanmaz; `data/r01/` yerel ve Git tarafından yok sayılan kontrollü extraction alanıdır.

## Veri seti

IPAD (Industrial Process Anomaly Detection Dataset), endüstriyel süreç videoları için yayımlanmış bir veri setidir. Bu çalışma `R01_conveyer_belt_real` senaryosunu kullanır. Kaynak: [IPAD dataset project page](https://github.com/bitvis/Industrial-Process-Anomaly-Detection). Makaledeki R01 için bildirilen AUC yaklaşık %84.4'tür; bu, bu projede elde edilmiş bir sonuç değildir.

## Yöntem planı

- **E0 — Frame baseline:** ImageNet ön-eğitimli ResNet18, normal eğitim frame embedding'lerinden bellek bankası, k-NN skoru, validation ile eşik. Testte AUROC, AUPR, F1, precision, recall, accuracy ve confusion counts.
- **E1 — Video baseline:** 16 frame, stride 4 klipler; dondurulmuş Video Swin Tiny özellikleri, normal klip bellek bankası; klip skorlarını frame zaman çizgisine geri eşleme.
- **E2 — Temporal model:** Video Swin özelliklerine Transformer Encoder veya GRU/LSTM.
- **E3 — İleri yöntemler:** Periodic memory, sliding window, reconstruction ve LoRA. Şimdilik uygulanmayacak.

## Kurulum

Python 3.10+ önerilir. macOS Apple Silicon'da PyTorch MPS, ardından CUDA, ardından CPU otomatik seçilir.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch kurulum seçenekleri platforma göre değişebildiğinden gerekirse [PyTorch'un resmi kurulum seçicisini](https://pytorch.org/get-started/locally/) kullanın.

## Dataset hazırlama ve inceleme

Çıkarılmış kopya `/Users/yusuf/Downloads/IPAD_dataset/R01` konumunda bulundu ve proje içindeki `data/r01` bu klasöre sembolik bağlantıdır; veri seti proje içine kopyalanmadı. Yapı `training/frames/NN`, `testing/frames/NN` ve `test_label/NNN.npy` şeklindedir. Arşivden yeniden çıkarmak gerekirse yalnızca R01 klasörünü çıkarın:

```bash
python scripts/extract_r01.py --zip /path/to/ipad_dataset.zip
python scripts/inspect_dataset.py --root data/r01
python scripts/prepare_r01.py --root data/r01
```

`inspect_dataset.py` gerçek klasörleri, frame adlandırmasını, çözünürlükleri, etiket kaynaklarını ve sayımları raporlar. `prepare_r01.py` test label vektörlerini framelere eşler ve eğitim sequence’lerini train/validation arasında sequence bazında ayırır. `data/README.md` ve `docs/DATASET.md` ayrıntıları içerir.

## Eğitim ve değerlendirme

### Google Colab Training (önerilen)

Model çalıştırmaları için önerilen ortam Google Colab GPU'dur. Proje klasörünü Google Drive'a `/content/drive/MyDrive/bitirme` yoluna yükleyin. Yerel dataset şu konumdadır: `/Users/yusuf/Downloads/IPAD_dataset/R01`; bu `R01` klasörünün tamamını Drive'da `MyDrive/bitirme_dataset/R01` konumuna yükleyin (yaklaşık 311 MB). Yeni dataset indirme gerekmez.

1. Proje klasörünü `MyDrive/bitirme` içine yükleyin.
2. `R01` dataset klasörünü `MyDrive/bitirme_dataset/R01` içine yükleyin.
3. Colab'da **Runtime → Change runtime type → T4 GPU** seçin.
4. Proje içindeki `notebooks/E0_R01_Frame_Baseline_Colab.ipynb` notebook'unu Colab'da açın.
5. İlk Drive mount hücresinden başlayarak hücreleri sırayla çalıştırın. Yapılandırma hücresindeki `PROJECT_ROOT` ve `DATASET_ROOT` yollarını gerektiğinde güncelleyin. Eğitim/threshold/test adımları Colab'da çalışır ve run dosyaları Drive'a kaydolur.

`notebooks/E1_R01_Video_Swin_Colab.ipynb` Colab'da önce bir train/validation/test sequence'iyle smoke test, sonra kullanıcı açıkça hücreyi açtığında full feature extraction ve test değerlendirmesini çalıştırır. Yerel Mac'i dataset inceleme, kod geliştirme ve smoke test için kullanın; full E1 için Colab GPU önerilir. Pipeline, torchvision Video Swin Tiny pretrained Kinetics-400 ağırlıklarını ve resmi deterministic transform'u kullanır. Klip cache'leri `models/E1_video_swin16_cache/` altında saklanır.

### Local E0 commands

```bash
python -m src.training.train_frame_baseline --config configs/frame_baseline.yaml
python -m src.evaluation.evaluate_frame --run-dir models/<E0-run>
```

**Because IPAD training data contains only normal operation, the anomaly threshold is calibrated exclusively from held-out normal validation sequences.** E0 uses frozen ImageNet ResNet18 512-dimensional global average pooled features from 256×256 inputs. Training uses normal R01 training sequences; validation uses held-out normal training sequences, split by sequence with seed 42. The default threshold is the 99.5th percentile of validation normal anomaly scores. Official testing sequences and `.npy` frame labels are used only for final evaluation; test data never tunes the threshold. Each run saves config, threshold statistics, memory bank, test predictions, metrics, and a run summary under `models/E0_frame_resnet18_<timestamp>/`. E1 iskeleti:

```bash
python -m scripts.smoke_test_video --batch-size 1
python -m src.training.train_video_baseline --config configs/video_swin.yaml --batch-size 4
python -m src.evaluation.evaluate_video --run-dir models/<E1-run>
```

Video Swin ağırlıklarını timm model deposundan ilk çalıştırmada edinmek gerekebilir. Uzun eğitim otomatik başlamaz.

## Streamlit

```bash
streamlit run app/streamlit_app.py
```

İlk sürüm model veya dataset olmadan açılır; video yükleme, önizleme, FPS, frame sayısı, çözünürlük ve süreyi gösterir. Model koşusu eklenince anomaly timeline ve segmentler gösterilebilir.

## Sonuçların durumu

E0 final test sonucu tamamlandı. E1 pipeline kodu ve sınırlı smoke test hazır; full E1 ve final test metrikleri henüz üretilmedi. Yerel smoke test run'ı performans sonucu değildir. Sonuçlar her full koşunun kendi `metrics.json` ve tahmin dosyalarından raporlanmalıdır.
