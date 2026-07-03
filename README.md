# TagAI - Açık Kaynak Türkçe Yapay Zeka

**TagAI**, tamamen Türkçe odaklı, açık kaynak, topluluk destekli bir büyük dil modelidir (LLM).

## 🎯 Hedefler

- ✅ Doğal ve akıcı Türkçe konuşma
- ✅ Kod yazma ve açıklama
- ✅ Mantık yürütme ve problem çözme
- ✅ Teknik ve genel bilgi
- ✅ Günlük asistan kullanımı
- ✅ Sürekli geliştirilebilir yapı

## 🏗️ Mimari

- **Base Model**: Açık kaynak foundation model (Llama, Mistral, Qwen vb.)
- **Eğitim Yöntemi**: QLoRA + PEFT (Parameter-Efficient Fine-Tuning)
- **Eğitim Ortamı**: Google Colab (T4/L4/A100 GPU)
- **Depolama**: 15GB Google Drive limiti içinde optimize edilmiş
- **Veri Seti**: Özel olarak oluşturulmuş yüksek kaliteli Türkçe veri

## 📊 Veri Seti

Veri setimiz aşağıdaki kategorileri içerir:

### Teknik İçerik
- Yazılım geliştirme (Python, JavaScript, C++, Java, Rust, Go vb.)
- Web geliştirme (React, Next.js, Node.js, HTML/CSS)
- DevOps (Docker, Kubernetes, Linux, Git)
- Veri bilimi ve makine öğrenmesi
- Algoritmalar ve veri yapıları
- Siber güvenlik

### Genel Bilgi
- Matematik, fizik, kimya, biyoloji
- Tarih, coğrafya, ekonomi
- Sağlık, teknoloji, oyun
- Günlük konuşma ve diyaloglar

### Yetenekler
- Kod yazma ve açıklama
- Problem çözme
- Metin düzeltme ve özet çıkarma
- Dokümantasyon yazma
- Mantık soruları

## 📁 Proje Yapısı

```
TagAI/
├── dataset/              # Veri setleri
│   ├── raw/             # Ham veri
│   ├── processed/       # İşlenmiş veri
│   └── synthetic/       # Sentetik üretilmiş veri
├── training/            # Eğitim scriptleri
│   ├── train.py
│   ├── utils.py
│   └── config/
├── evaluation/          # Model değerlendirme
├── scripts/             # Yardımcı scriptler
├── notebooks/           # Colab notebook'ları
└── docs/               # Dokümantasyon
```

## 🚀 Hızlı Başlangıç

### 1. Ortam Hazırlığı

```bash
pip install -r requirements.txt
```

### 2. Google Colab'da Eğitim

1. `notebooks/TagAI_Training.ipynb` dosyasını Google Colab'da açın
2. GPU runtime'ı aktif edin (T4/L4/A100)
3. Google Drive'ı bağlayın
4. Notebook'taki adımları takip edin

### 3. Veri Seti Oluşturma

```bash
python scripts/data_collection.py
```

## ⚙️ Özellikler

### Bellek Optimizasyonu
- QLoRA ile 4-bit quantization
- Gradient checkpointing
- Flash Attention 2 desteği
- Mixed precision training (bf16/fp16)

### Depolama Yönetimi
- Otomatik checkpoint temizleme
- Yalnızca son 2-3 checkpoint saklama
- En iyi model ayrıca korunur
- Geçici dosya otomatik temizleme
- 15GB Google Drive limiti içinde çalışma

### Eğitim Kararlılığı
- Kesintiye dayanıklı checkpoint sistemi
- Otomatik devam etme
- Hata kurtarma mekanizmaları
- GPU kullanım izleme

## 📈 Performans

- **GPU**: T4 üzerinde ~1-2 epoch/saat (model boyutuna bağlı)
- **Bellek**: ~12-15GB VRAM kullanımı
- **Depolama**: ~10-12GB checkpoint + veri

## 🔧 Geliştirme

### Veri Seti Ekleme

```python
from scripts.data_collection import add_custom_data

add_custom_data(
    category="özel_kategori",
    data_path="your_data.json"
)
```

### Model Değerlendirme

```python
python evaluation/evaluate.py --checkpoint models/checkpoint-1000
```

## 🤝 Katkıda Bulunma

Bu proje açık kaynak ve topluluk desteklidir. Katkılarınızı bekliyoruz!

1. Fork yapın
2. Yeni branch oluşturun (`git checkout -b feature/yeni-ozellik`)
3. Değişikliklerinizi commit edin
4. Push yapın
5. Pull Request oluşturun

## 📝 Lisans

Bu proje MIT lisansı altında sunulmaktadır.

## 🙏 Teşekkürler

- Hugging Face topluluğu
- Google Colab ekibi
- Tüm açık kaynak katkıcılar

## 📧 İletişim

Sorularınız ve önerileriniz için issue açabilirsiniz.

---

**Not**: Bu proje eğitim ve araştırma amaçlıdır. Üretim ortamında kullanmadan önce kapsamlı testler yapılmalıdır.
