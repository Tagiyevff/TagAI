"""
TagAI - Model Değerlendirme Scripti
Eğitilmiş modeli test et ve performansını ölç
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
from pathlib import Path
from typing import List, Dict, Optional
import logging
from tqdm import tqdm

from metrics import compute_all_metrics, TurkishTextMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Model değerlendirme sınıfı"""

    def __init__(self, model_path: str, base_model: Optional[str] = None):
        """
        Args:
            model_path: Eğitilmiş model yolu (LoRA adaptörleri)
            base_model: Base model adı (opsiyonel, model_path'ten çıkarılabilir)
        """
        self.model_path = model_path
        self.base_model = base_model
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Modeli yükle"""
        logger.info("🤖 Model yükleniyor...")

        # Base model belirle
        if self.base_model is None:
            config_path = Path(self.model_path) / "adapter_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self.base_model = config.get('base_model_name_or_path')

        if self.base_model is None:
            raise ValueError("Base model belirtilmedi ve tespit edilemedi!")

        logger.info(f"Base model: {self.base_model}")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Base model yükle
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            device_map="auto",
            torch_dtype=torch.float16
        )

        # LoRA adaptörlerini yükle
        self.model = PeftModel.from_pretrained(self.model, self.model_path)

        self.model.eval()

        logger.info("✅ Model yüklendi")

    def generate_response(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Prompt'a yanıt üret

        Args:
            prompt: Giriş metni
            max_length: Maksimum üretim uzunluğu
            temperature: Sampling temperature
            top_p: Nucleus sampling

        Returns:
            Üretilen metin
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Prompt kısmını çıkar
        if response.startswith(prompt):
            response = response[len(prompt):].strip()

        return response

    def evaluate_test_cases(self, test_cases: List[Dict]) -> Dict:
        """
        Test durumlarını değerlendir

        Args:
            test_cases: [{"prompt": "...", "expected": "...", "category": "..."}]

        Returns:
            Değerlendirme sonuçları
        """
        logger.info(f"📊 {len(test_cases)} test durumu değerlendiriliyor...")

        results = []
        turkish_metrics = TurkishTextMetrics()

        for i, test_case in enumerate(tqdm(test_cases, desc="Değerlendirme")):
            prompt = test_case['prompt']
            expected = test_case.get('expected', '')
            category = test_case.get('category', 'general')

            # Yanıt üret
            response = self.generate_response(prompt)

            # Metrikler
            quality_score = (
                turkish_metrics.check_turkish_characters(response) * 0.3 +
                turkish_metrics.check_sentence_structure(response) * 0.4 +
                turkish_metrics.check_coherence(response) * 0.3
            )

            result = {
                'id': i,
                'category': category,
                'prompt': prompt,
                'expected': expected,
                'generated': response,
                'turkish_quality': quality_score,
                'length': len(response.split())
            }

            results.append(result)

        # Özet istatistikler
        summary = self._compute_summary(results)

        return {
            'results': results,
            'summary': summary
        }

    def _compute_summary(self, results: List[Dict]) -> Dict:
        """Özet istatistikler hesapla"""
        summary = {
            'total_tests': len(results),
            'avg_quality': sum(r['turkish_quality'] for r in results) / len(results),
            'avg_length': sum(r['length'] for r in results) / len(results),
            'by_category': {}
        }

        # Kategorilere göre
        categories = set(r['category'] for r in results)
        for cat in categories:
            cat_results = [r for r in results if r['category'] == cat]
            summary['by_category'][cat] = {
                'count': len(cat_results),
                'avg_quality': sum(r['turkish_quality'] for r in cat_results) / len(cat_results),
                'avg_length': sum(r['length'] for r in cat_results) / len(cat_results)
            }

        return summary

    def print_evaluation_report(self, evaluation: Dict):
        """Değerlendirme raporunu yazdır"""
        summary = evaluation['summary']

        print("\n" + "="*60)
        print("📊 DEĞERLENDİRME RAPORU")
        print("="*60)
        print(f"Toplam Test: {summary['total_tests']}")
        print(f"Ortalama Kalite: {summary['avg_quality']:.2%}")
        print(f"Ortalama Uzunluk: {summary['avg_length']:.1f} kelime")

        print("\n📋 Kategorilere Göre:")
        for cat, stats in summary['by_category'].items():
            print(f"\n  {cat.upper()}:")
            print(f"    Test Sayısı: {stats['count']}")
            print(f"    Kalite: {stats['avg_quality']:.2%}")
            print(f"    Uzunluk: {stats['avg_length']:.1f} kelime")

        print("\n" + "="*60)

        # Örnek çıktılar (ilk 3)
        print("\n📝 ÖRNEK ÇIKTILAR (İlk 3):\n")
        for i, result in enumerate(evaluation['results'][:3]):
            print(f"--- Örnek {i+1} ({result['category']}) ---")
            print(f"Prompt: {result['prompt'][:100]}...")
            print(f"Üretilen: {result['generated'][:200]}...")
            print(f"Kalite: {result['turkish_quality']:.2%}")
            print()

    def save_results(self, evaluation: Dict, output_path: str):
        """Sonuçları kaydet"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ Sonuçlar kaydedildi: {output_path}")


def create_test_cases() -> List[Dict]:
    """Örnek test durumları oluştur"""
    return [
        # Genel konuşma
        {
            'prompt': 'Merhaba! Nasılsın?',
            'category': 'conversation',
            'expected': 'Greeting response'
        },
        {
            'prompt': 'Türkiye\'nin başkenti neresidir?',
            'category': 'knowledge',
            'expected': 'Ankara'
        },

        # Kod yazma
        {
            'prompt': 'Python\'da liste elemanlarını ters çevir',
            'category': 'coding',
            'expected': 'list.reverse() or [::-1]'
        },
        {
            'prompt': 'React\'ta useState hook\'u nasıl kullanılır?',
            'category': 'coding',
            'expected': 'useState example'
        },

        # Mantık
        {
            'prompt': 'Eğer A>B ve B>C ise, A ve C arasındaki ilişki nedir?',
            'category': 'reasoning',
            'expected': 'A>C (transitive property)'
        },

        # Bilgi
        {
            'prompt': 'Fotosentez nedir ve nasıl gerçekleşir?',
            'category': 'knowledge',
            'expected': 'Photosynthesis explanation'
        },
    ]


def main():
    """Ana fonksiyon"""
    import argparse

    parser = argparse.ArgumentParser(description="TagAI Model Değerlendirme")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/TagAI-final",
        help="Model yolu"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base model (opsiyonel)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/results.json",
        help="Sonuç kayıt yolu"
    )

    args = parser.parse_args()

    # Evaluator oluştur
    evaluator = ModelEvaluator(args.model_path, args.base_model)
    evaluator.load_model()

    # Test durumları
    test_cases = create_test_cases()

    # Değerlendirme
    evaluation = evaluator.evaluate_test_cases(test_cases)

    # Rapor
    evaluator.print_evaluation_report(evaluation)

    # Kaydet
    evaluator.save_results(evaluation, args.output)

    print("\n✅ Değerlendirme tamamlandı!")


if __name__ == "__main__":
    main()
