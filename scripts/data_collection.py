"""
TagAI - Veri Toplama ve Üretimi
Türkçe veri setini oluşturmak için çoklu kaynaklardan veri toplar
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCollector:
    """Veri toplama ve organizasyon için ana sınıf"""

    def __init__(self, output_dir: str = "dataset/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {
            'total_samples': 0,
            'by_category': {}
        }

    def save_dataset(self, data: List[Dict], filename: str, category: str):
        """Veri setini JSON formatında kaydet"""
        output_path = self.output_dir / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # İstatistikleri güncelle
        count = len(data)
        self.stats['total_samples'] += count
        self.stats['by_category'][category] = self.stats['by_category'].get(category, 0) + count

        logger.info(f"✅ {count} örnek kaydedildi: {filename} ({category})")

    def print_statistics(self):
        """Toplama istatistiklerini yazdır"""
        print("\n" + "="*60)
        print("📊 VERİ TOPLAMA İSTATİSTİKLERİ")
        print("="*60)
        print(f"Toplam Örnek Sayısı: {self.stats['total_samples']:,}")
        print("\nKategorilere Göre Dağılım:")
        for category, count in sorted(self.stats['by_category'].items()):
            percentage = (count / self.stats['total_samples']) * 100
            print(f"  {category}: {count:,} ({percentage:.1f}%)")
        print("="*60 + "\n")


class SyntheticDataGenerator:
    """Sentetik Türkçe veri üreteci"""

    def __init__(self):
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict:
        """Veri üretimi için şablonları yükle"""
        return {
            'conversation': self._conversation_templates(),
            'coding': self._coding_templates(),
            'reasoning': self._reasoning_templates(),
            'knowledge': self._knowledge_templates(),
            'text_processing': self._text_processing_templates(),
        }

    def _conversation_templates(self) -> List[Dict]:
        """Günlük konuşma şablonları"""
        return [
            {
                'prompt': 'Merhaba! Nasılsın?',
                'response': 'Merhaba! Ben TagAI, Türkçe konuşabilen bir yapay zeka asistanıyım. Size nasıl yardımcı olabilirim?'
            },
            {
                'prompt': 'Bugün ne yapmalıyım?',
                'response': 'Size özel öneriler verebilmem için daha fazla bilgiye ihtiyacım var. Ne tür aktiviteler ilginizi çeker? İş, eğlence, spor veya başka bir şey mi düşünüyorsunuz?'
            },
            {
                'prompt': 'Türkçe dilbilgisi kurallarını açıklar mısın?',
                'response': 'Türkçe, sondan eklemeli bir dildir. Temel dilbilgisi özellikleri:\n\n1. Ekler: Türkçede kelimeler köke ekler getirilerek türetilir\n2. Ünlü uyumu: Ekler kelimenin son ünlüsüne göre değişir\n3. Cümle yapısı: Özne-Nesne-Yüklem sırası kullanılır\n\nDaha spesifik bir konu hakkında bilgi ister misiniz?'
            }
        ]

    def _coding_templates(self) -> List[Dict]:
        """Kod yazma şablonları"""
        return [
            {
                'prompt': 'Python\'da liste oluşturma',
                'response': 'Python\'da liste oluşturmanın birkaç yolu vardır:\n\n```python\n# Boş liste\nliste = []\n\n# Değerlerle liste\nsayilar = [1, 2, 3, 4, 5]\n\n# List comprehension\nkareler = [x**2 for x in range(10)]\n\n# Liste metodları\nliste.append(10)  # Eleman ekle\nliste.extend([11, 12])  # Birden fazla ekle\n```'
            },
            {
                'prompt': 'React\'ta component nasıl yazılır?',
                'response': 'React\'ta functional component örneği:\n\n```jsx\nimport React from \'react\';\n\nfunction MyComponent({ title, children }) {\n  return (\n    <div className="component">\n      <h1>{title}</h1>\n      <div>{children}</div>\n    </div>\n  );\n}\n\nexport default MyComponent;\n```\n\nKullanımı:\n```jsx\n<MyComponent title="Merhaba">\n  <p>İçerik buraya</p>\n</MyComponent>\n```'
            }
        ]

    def _reasoning_templates(self) -> List[Dict]:
        """Mantık yürütme şablonları"""
        return [
            {
                'prompt': 'Eğer bir sayı 2 ve 3 ile bölünebiliyorsa, 6 ile de bölünebilir mi?',
                'response': 'Evet, bu matematiksel olarak doğrudur.\n\nMantık:\n1. Bir sayı hem 2 hem de 3 ile tam bölünebiliyorsa\n2. O sayı 2 ve 3\'ün ortak katıdır\n3. 2 ve 3 asal sayılardır ve aralarında asaldır\n4. Bu durumda sayı 2×3 = 6 ile de bölünebilir\n\nÖrnek: 12 sayısı hem 2 hem 3 ile bölünür, ve 6 ile de bölünür (12÷6=2)'
            }
        ]

    def _knowledge_templates(self) -> List[Dict]:
        """Bilgi soruları şablonları"""
        return [
            {
                'prompt': 'Fotosentez nedir?',
                'response': 'Fotosentez, bitkilerin ışık enerjisini kimyasal enerjiye dönüştürdüğü yaşamsal bir süreçtir.\n\n**Süreç:**\n1. Bitkiler klorofil pigmenti ile güneş ışığını yakalar\n2. Su (H₂O) ve karbondioksit (CO₂) alınır\n3. Işık enerjisi kullanılarak glikoz (C₆H₁₂O₆) üretilir\n4. Yan ürün olarak oksijen (O₂) açığa çıkar\n\n**Formül:** 6CO₂ + 6H₂O + ışık → C₆H₁₂O₆ + 6O₂\n\nBu süreç atmosferdeki oksijen kaynağının temelidir.'
            }
        ]

    def _text_processing_templates(self) -> List[Dict]:
        """Metin işleme şablonları"""
        return [
            {
                'prompt': 'Bu metni özetle: [uzun metin]',
                'response': 'Özet çıkarmak için metni analiz edip ana fikirleri belirliyorum. Lütfen özetlemek istediğiniz metni paylaşın.'
            }
        ]

    def generate_samples(self, category: str, count: int) -> List[Dict]:
        """Belirtilen kategoride sentetik veri üret"""
        if category not in self.templates:
            logger.warning(f"Kategori bulunamadı: {category}")
            return []

        templates = self.templates[category]
        samples = []

        for i in range(count):
            template = random.choice(templates)
            sample = {
                'id': f"{category}_{i}",
                'category': category,
                'instruction': template['prompt'],
                'response': template['response'],
                'metadata': {
                    'source': 'synthetic',
                    'generated_at': datetime.now().isoformat()
                }
            }
            samples.append(sample)

        return samples


def create_base_dataset():
    """Temel veri setini oluştur"""
    collector = DataCollector()
    generator = SyntheticDataGenerator()

    print("\n🚀 Veri seti oluşturma başlatılıyor...\n")

    # Her kategori için sentetik veri üret
    categories = {
        'conversation': 500,
        'coding': 1000,
        'reasoning': 300,
        'knowledge': 500,
        'text_processing': 200,
    }

    for category, count in categories.items():
        logger.info(f"📝 {category} kategorisi için {count} örnek üretiliyor...")
        samples = generator.generate_samples(category, count)
        collector.save_dataset(samples, f"{category}.json", category)

    collector.print_statistics()

    print("\n✅ Temel veri seti oluşturuldu!")
    print("📁 Konum: dataset/raw/")
    print("\n⚠️  NOT: Bu temel bir başlangıç setidir.")
    print("Daha kaliteli ve kapsamlı veri için ek kaynaklar eklenmelidir.")


if __name__ == "__main__":
    create_base_dataset()
