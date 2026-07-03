# TagAI - Veri Stratejisi

## 🎯 Hedef

Yüksek kaliteli, çeşitli ve dengeli bir Türkçe veri seti oluşturarak TagAI'yi güçlü bir dil modeli haline getirmek.

## 📊 Veri Kategori Dağılımı

| Kategori | Oran | Min. Örnek | Açıklama |
|----------|------|------------|----------|
| Kod Yazma | %25 | 5,000 | Python, JS, web dev, algoritma |
| Teknik QA | %20 | 4,000 | Yazılım, DevOps, sistem soruları |
| Genel Konuşma | %15 | 3,000 | Günlük diyalog, selamlaşma |
| Bilgi | %15 | 3,000 | Matematik, fen, tarih, genel kültür |
| Metin İşleme | %15 | 3,000 | Özet, düzeltme, dönüştürme |
| Mantık Yürütme | %10 | 2,000 | Problem çözme, analitik düşünme |

**Toplam Hedef:** ~20,000+ örnek (başlangıç için)

## 🔄 Veri Toplama Stratejisi

### Faz 1: Temel Sentetik Veri (Şu an)
- Template tabanlı veri üretimi
- Her kategori için temel örnekler
- Hızlı prototip için yeterli

**장점:**
- ✅ Hızlı başlangıç
- ✅ Kalite kontrolü kolay
- ✅ Telif sorunu yok

**Eksileri:**
- ⚠️ Sınırlı çeşitlilik
- ⚠️ Template'e bağımlılık

### Faz 2: Açık Kaynak Veri (Planlanan)

**Kaynaklar:**
1. **Wikipedia TR**: Bilgi ve genel kültür
2. **Turkish NLP Datasets**: Akademik veri setleri
3. **GitHub**: Türkçe kommentli kod örnekleri
4. **Stack Overflow**: Türkçe teknik sorular (uygunsa)

**İşleme Adımları:**
```python
1. İndir → 2. Temizle → 3. Formatla → 4. Filtrele → 5. Birleştir
```

### Faz 3: LLM ile Veri Genişletme (Planlanan)

Mevcut güçlü modeller (GPT-4, Claude vb.) kullanarak:
- Var olan örnekleri çeşitlendirme
- Yeni senaryolar üretme
- Kalite kontrolü

**Örnek Workflow:**
```
Template → LLM genişletme → İnsan doğrulama → Veri setine ekle
```

## 🧹 Veri Temizleme Pipeline

### 1. Format Kontrolü
```python
- JSON geçerliliği
- Zorunlu alanlar (instruction, response)
- Karakter encoding (UTF-8)
```

### 2. Kalite Filtreleri

**Uzunluk Filtreleri:**
- Minimum: 10 karakter
- Maksimum: 4096 karakter
- Boş alan kontrolü

**İçerik Filtreleri:**
- Bozuk karakter tespiti
- HTML/markup temizleme
- URL normalizasyonu (koru ama temizle)

**Tekrar Kontrolü:**
```python
# Exact match
if text in seen_texts:
    skip()

# Near duplicate (fuzzy)
if similarity(text, existing) > 0.95:
    skip()
```

**Dil Tespiti:**
```python
# Türkçe karakter oranı
turkish_chars = 'çğıöşüÇĞİÖŞÜ'
if turkish_ratio < 0.3:
    flag_for_review()
```

### 3. Dengeleme

**Kategori Dengelemesi:**
```python
target_distribution = {
    'coding': 0.25,
    'technical_qa': 0.20,
    # ...
}

# Undersampling veya oversampling
balance_dataset(data, target_distribution)
```

**Zorluk Dengelemesi:**
- Kolay: %40
- Orta: %40  
- Zor: %20

## 📈 Veri Kalite Metrikleri

### Otomatik Metrikler

1. **Çeşitlilik Skoru**
   ```python
   unique_words / total_words
   ```
   Hedef: >0.60

2. **Türkçe Kalitesi**
   ```python
   - Türkçe karakter kullanımı
   - Cümle yapısı
   - Kelime tekrarı
   ```
   Hedef: >0.80

3. **Yanıt Kalitesi**
   ```python
   - Yanıt uzunluğu / soru uzunluğu
   - Alakalılık (keyword match)
   - Tamamlık (cümle sonları)
   ```
   Hedef: >0.75

### Manuel Kontrol

**Örnekleme Stratejisi:**
- Her kategori için rastgele 50 örnek
- En yüksek/en düşük skorlu 20 örnek
- Sınır durumlar (edge cases)

**Kontrol Kriterleri:**
- ✅ Soru anlamlı mı?
- ✅ Yanıt doğru mu?
- ✅ Türkçe doğal mı?
- ✅ İlgili bilgi içeriyor mu?

## 🔧 Veri Üretim Araçları

### Template Engine

```python
class TemplateGenerator:
    def generate(self, category, count):
        templates = self.load_templates(category)
        samples = []
        
        for i in range(count):
            template = random.choice(templates)
            sample = self.fill_template(template)
            samples.append(sample)
        
        return samples
```

### Varyasyon Üreteci

```python
def create_variations(base_example, num_variations=5):
    """Bir örnekten varyasyonlar üret"""
    variations = []
    
    # Kelime değiştirme
    # Cümle yeniden düzenleme
    # Detay ekleme/çıkarma
    
    return variations
```

### Kalite Değerlendirici

```python
def evaluate_quality(sample):
    scores = {
        'length': check_length(sample),
        'turkish': check_turkish_quality(sample),
        'relevance': check_relevance(sample),
        'completeness': check_completeness(sample)
    }
    
    return weighted_average(scores)
```

## 📂 Veri Organizasyonu

### Dizin Yapısı
```
dataset/
├── raw/                    # Ham veri
│   ├── conversation.json
│   ├── coding.json
│   └── ...
├── processed/              # İşlenmiş veri
│   ├── train.json         # %95
│   └── validation.json    # %5
└── synthetic/              # Üretilen veri
    ├── templates/
    └── generated/
```

### Versiyon Kontrolü

```
dataset_v1.0.0/
├── metadata.json          # Versiyon bilgileri
├── statistics.json        # İstatistikler
├── train.json
└── validation.json
```

**metadata.json:**
```json
{
  "version": "1.0.0",
  "created_at": "2024-01-01",
  "total_samples": 20000,
  "sources": ["synthetic", "wikipedia"],
  "quality_score": 0.85
}
```

## 🚀 Uygulama Planı

### Şu An (Faz 1 - Tamamlandı)
- [x] Temel template sistemi
- [x] Sentetik veri üretimi
- [x] ~2,500 örnek (kategori başına 500)

### Yakın Vadede (Faz 2)
- [ ] Wikipedia TR entegrasyonu
- [ ] Açık kaynak veri toplama
- [ ] 20,000+ örneğe çıkarma
- [ ] Kalite kontrol pipeline

### Uzun Vadede (Faz 3)
- [ ] LLM destekli veri genişletme
- [ ] İnsan doğrulama sistemi
- [ ] 100,000+ örnek hedefi
- [ ] Sürekli veri iyileştirme

## 💡 İpuçları

1. **Kalite > Miktar**: 10,000 kaliteli örnek > 100,000 düşük kaliteli
2. **Çeşitlilik Önemli**: Aynı tipte çok örnek yerine farklı senaryolar
3. **Dengeleme**: Tüm kategorilere eşit önem
4. **Sürekli İyileştirme**: Model feedback'ine göre veri güncelle
5. **Doğrulama**: Veri setini periyodik olarak gözden geçir

## 📝 Sonraki Adımlar

1. Temel veri setini oluştur (`scripts/data_collection.py`)
2. Kalite metriklerini çalıştır
3. Manuel olarak örnekleri kontrol et
4. Eksik kategorileri belirle
5. Veri setini genişlet
6. Model eğitimi başlat
7. Model performansına göre veri setini iyileştir
