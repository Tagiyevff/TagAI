# TagAI - Veri Seti Dizini

Bu dizin TagAI modelinin eğitiminde kullanılacak Türkçe veri setlerini içerir.

## 📁 Dizin Yapısı

```
dataset/
├── raw/                # Ham veri (işlenmemiş)
├── processed/          # İşlenmiş ve temizlenmiş veri
└── synthetic/          # Sentetik üretilmiş veri
```

## 🎯 Veri Kategorileri

### 1. Genel Konuşma (conversation)
- Günlük diyaloglar
- Selamlaşma ve sohbet
- Soru-cevap

### 2. Teknik Sorular (technical_qa)
- Yazılım geliştirme soruları
- Teknik dokümantasyon
- Problem çözme

### 3. Kod Yazma (coding)
- Python, JavaScript, TypeScript, C++, Java, Rust, Go
- Web geliştirme (React, Next.js, Node.js)
- Kod açıklama ve inceleme

### 4. Mantık Yürütme (reasoning)
- Mantık problemleri
- Matematik soruları
- Analitik düşünme

### 5. Bilgi (knowledge)
- Matematik, fizik, kimya, biyoloji
- Tarih, coğrafya, ekonomi
- Genel kültür

### 6. Metin İşleme (text_processing)
- Özet çıkarma
- Metin düzeltme
- Çeviri
- Dokümantasyon

## 📊 Veri Formatı

Tüm veri JSON formatında saklanır:

```json
{
  "id": "unique_id",
  "category": "coding",
  "instruction": "Python'da liste oluştur",
  "response": "İşte Python'da liste oluşturma...",
  "metadata": {
    "source": "synthetic",
    "language": "tr",
    "created_at": "2024-01-01T00:00:00"
  }
}
```

## 🛠️ Veri İşleme

Veri işleme adımları:

1. **Toplama**: `scripts/data_collection.py`
2. **Temizleme**: Tekrarları kaldır, formatı düzelt
3. **Doğrulama**: Kalite kontrolleri
4. **Bölümleme**: Train/validation split

## 📈 Kalite Kriterleri

- ✅ Minimum 10 karakter
- ✅ Maksimum 4096 karakter
- ✅ Türkçe dil kontrolü
- ✅ Tekrar kontrolü
- ✅ Format doğrulama

## 🚀 Kullanım

```python
from scripts.data_collection import create_base_dataset

# Temel veri seti oluştur
create_base_dataset()
```

## 📝 Notlar

- Veri setleri `dataset/raw/` altında kategori bazlı JSON dosyaları olarak saklanır
- İşlenmiş veri `dataset/processed/` altında birleştirilir
- Google Drive'da alan tasarrufu için gereksiz dosyalar silinir
