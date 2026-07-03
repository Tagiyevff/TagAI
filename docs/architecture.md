# TagAI - Proje Mimarisi

## 🏗️ Genel Mimari

TagAI, açık kaynak bir foundation model üzerine QLoRA/PEFT ile Türkçe'ye özelleştirilmiş bir dil modelidir.

```
┌─────────────────────────────────────────────┐
│         Base Model (Llama/Mistral/Qwen)     │
│              (4-bit Quantized)              │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│            LoRA Adaptörler                  │
│         (Eğitilebilir Parametreler)         │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│              TagAI Model                    │
│          (Türkçe Optimized)                 │
└─────────────────────────────────────────────┘
```

## 📦 Modül Yapısı

### 1. Veri Katmanı (`dataset/`)
- **raw/**: Ham veri dosyaları (JSON)
- **processed/**: İşlenmiş ve birleştirilmiş veri
- **synthetic/**: Sentetik üretilmiş veri

**Sorumluluk**: Veri toplama, temizleme, formatlamaç

### 2. Eğitim Katmanı (`training/`)
- **train.py**: Ana eğitim scripti
- **utils.py**: Yardımcı fonksiyonlar
- **config/**: Eğitim konfigürasyonları

**Sorumluluk**: Model eğitimi, checkpoint yönetimi

### 3. Değerlendirme Katmanı (`evaluation/`)
- **evaluate.py**: Model test scripti
- **metrics.py**: Performans metrikleri

**Sorumluluk**: Model kalite ölçümü

### 4. Yardımcı Scriptler (`scripts/`)
- **setup_colab.py**: Ortam hazırlığı
- **manage_storage.py**: Depolama yönetimi
- **data_collection.py**: Veri üretimi

**Sorumluluk**: Altyapı ve otomasyonlar

## 🔄 Eğitim Akışı

```
1. Ortam Hazırlığı
   ├─ GPU tespiti
   ├─ Kütüphane kurulumu
   └─ Google Drive bağlantısı

2. Veri Hazırlığı
   ├─ Veri toplama/üretme
   ├─ Temizleme ve formatlama
   └─ Train/validation split

3. Model Yükleme
   ├─ Base model (4-bit quantization)
   ├─ LoRA konfigürasyonu
   └─ PEFT adaptasyonu

4. Eğitim
   ├─ QLoRA ile fine-tuning
   ├─ Gradient checkpointing
   ├─ Mixed precision training
   └─ Checkpoint kaydetme

5. Değerlendirme
   ├─ Validation metrikleri
   ├─ Test case çalıştırma
   └─ Kalite analizi

6. Model Kaydetme
   ├─ LoRA adaptörlerini kaydet
   ├─ Tokenizer'ı kaydet
   └─ Model card oluştur
```

## ⚙️ Teknik Detaylar

### QLoRA (Quantized Low-Rank Adaptation)

**Neden QLoRA?**
- %95 parametre tasarrufu
- GPU bellek kullanımını 4-8x azaltır
- Ücretsiz Colab T4'te eğitim mümkün

**Nasıl Çalışır?**
```
Base Model (Frozen, 4-bit)
    ├─ q_proj + LoRA
    ├─ k_proj + LoRA
    ├─ v_proj + LoRA
    └─ ...

Sadece LoRA parametreleri eğitilir (~1-2% toplam)
```

### Bellek Optimizasyonu

1. **4-bit Quantization**: Model ağırlıkları 4-bit'te
2. **Gradient Checkpointing**: Aktivasyonlar yeniden hesaplanır
3. **Paged Optimizer**: CPU/GPU bellek swap
4. **Mixed Precision**: BF16/FP16 hesaplama

### Depolama Yönetimi

**15GB Google Drive Limiti İçin Stratejiler:**

1. **Checkpoint Rotasyonu**
   - Sadece son 2 checkpoint saklanır
   - En iyi model ayrıca korunur
   - Eskiler otomatik silinir

2. **Sıkıştırma**
   - Sadece LoRA ağırlıkları kaydedilir
   - Optimizer state opsiyonel
   - Gereksiz metadata çıkarılır

3. **Otomatik Temizlik**
   - Geçici dosyalar silinir
   - Cache temizlenir
   - Boş alan izlenir

## 🔧 Konfigürasyon Sistemi

### Hiyerarşi

```
config.yaml (global)
    └─ training/config/training_config.yaml (auto-generated)
        └─ GPU-specific overrides (runtime)
```

### GPU Adaptasyonu

**T4 GPU:**
- Batch size: 2
- Gradient accumulation: 4
- FP16 precision
- Effective batch: 8

**A100 GPU:**
- Batch size: 8
- Gradient accumulation: 1
- BF16 precision
- Effective batch: 8

## 📊 Veri Formatı

### Alpaca Format
```json
{
  "instruction": "Kullanıcı talimatı",
  "response": "Model yanıtı",
  "category": "coding|reasoning|...",
  "metadata": {...}
}
```

### Tokenization
```
### Talimat:
{instruction}

### Yanıt:
{response}
```

## 🚀 Genişletilebilirlik

### Yeni Veri Kategorisi Ekleme

1. `scripts/data_collection.py` içinde template ekle
2. `config.yaml` içinde kategori oranı belirt
3. Veri setini yeniden oluştur

### Farklı Base Model Kullanma

1. `config.yaml` içinde `base_model` değiştir
2. LoRA `target_modules` kontrol et (model mimarisine göre)
3. Training config'i güncelle

### Custom Metrik Ekleme

1. `evaluation/metrics.py` içinde metrik fonksiyonu yaz
2. `compute_all_metrics()` fonksiyonuna ekle
3. Evaluation script'te kullan

## 🔐 Güvenlik ve Kalite

### Veri Kalite Kontrolleri
- Minimum/maksimum uzunluk filtreleri
- Tekrar tespiti
- Dil tespiti (Türkçe)
- Format doğrulama

### Model Güvenliği
- Checkpoint imza kontrolü
- Güvenli model yükleme
- Input sanitization

## 📈 Performans Hedefleri

- **Eğitim Hızı**: ~1-2 epoch/saat (T4, 7B model)
- **Bellek Kullanımı**: <15GB VRAM
- **Depolama**: <12GB toplam
- **Inference Hızı**: ~10-20 token/saniye (T4)

## 🔄 Versiyon Yönetimi

- **Model**: Semantic versioning (v1.0.0)
- **Checkpoints**: Step numarası + timestamp
- **Config**: Hash tabanlı değişiklik takibi
