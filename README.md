# Drone görüntülerinde insan tespiti — YOLO ve VisDrone

VisDrone görüntülerinde insan tespiti üzerine yaptığım bilgisayarlı görü çalışması. VisDrone'un `pedestrian` ve `people` sınıflarını tek bir insan sınıfında birleştirip YOLOv8n ile bir eğitim denemesi yaptım. Bu repo veri dönüşüm kodunu ve o denemenin kayıtlarını içeriyor.

## Bu repoda ne var?

- [convert_visdrone_to_yolo.py](convert_visdrone_to_yolo.py): piksel koordinatlarını normalize YOLO kutularına dönüştürüyor. Yalnız 1 ve 2 numaralı sınıfları alıyor; insan içermeyen görüntüleri çıktı setine koymuyor.
- [results/args.yaml](results/args.yaml): kaydedilmiş eğitim ayarları. YOLOv8n, 50 epoch, batch 16 ve 640 piksel görüntü boyutu kullanılmış.
- [results/results.csv](results/results.csv): epoch bazında kayıp ve değerlendirme sonuçları.
- `results/weights/`: bu denemeye ait `best.pt` ve `last.pt` ağırlıkları.

## Kaydedilmiş sonuç

Aşağıdaki değerler `results.csv` içindeki **50. epoch** satırından geliyor; ayrı bir test seti sonucu veya makale sonucu olarak sunmuyorum.

| Ölçüt | Değer |
|---|---:|
| Precision | 0.61758 |
| Recall | 0.41878 |
| mAP@50 | 0.46723 |
| mAP@50–95 | 0.18720 |

Özellikle küçük ve kalabalık insan görüntülerinde kaçırılan tespitler var. Tek eğitim kaydı üzerinden genelleme veya aşırı öğrenme hakkında kesin sonuç çıkarmıyorum.

## Veri dönüşümünü çalıştırma

```bash
python -m venv .venv
python -m pip install opencv-python tqdm
python convert_visdrone_to_yolo.py
```

Kurulumu sanal ortamı etkinleştirdikten sonra yap. Ham veriyi [VisDrone'un kendi deposundan](https://github.com/VisDrone/VisDrone-Dataset) edinip köke `VisDrone2019-DET-train/` ve `VisDrone2019-DET-val/` klasörleriyle yerleştir. Kod `images/` ve `annotations/` alt klasörlerini bekliyor; çıktı `datasets/visdrone_human/` altında oluşuyor. Ham veri repoya dahil değil.

Eğitim kaydı Google Colab'daki `/content/data.yaml` dosyasına referans veriyor. Bu YAML ve eğitim notebook'u repoda bulunmadığından repo tek başına o koşuyu birebir yeniden üretmiyor. Ağırlık dosyalarının da yalnız güvenilen kaynaklardan yüklenmesi gerekir.

## Çalışmanın kapsamı

Burada mevcut kodu ve kayıtlı denemeyi paylaşıyorum. Bu README makalenin kabul edildiği veya yayımlandığı anlamına gelmiyor.
