"""
TagAI - Değerlendirme Metrikleri
Model performansını ölçmek için metrikler
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from transformers import PreTrainedTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """Model değerlendirme metrikleri"""

    def __init__(self, tokenizer: Optional[PreTrainedTokenizer] = None):
        self.tokenizer = tokenizer

    def compute_perplexity(self, loss: float) -> float:
        """
        Perplexity hesapla
        Perplexity = e^(loss)
        """
        try:
            perplexity = np.exp(loss)
            return float(perplexity)
        except OverflowError:
            return float('inf')

    def compute_accuracy(self, predictions: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Token düzeyinde doğruluk hesapla

        Args:
            predictions: Model tahminleri [batch_size, seq_len, vocab_size]
            labels: Gerçek değerler [batch_size, seq_len]
        """
        # En yüksek olasılıklı token'ları al
        pred_tokens = torch.argmax(predictions, dim=-1)

        # Padding token'larını maskeleme (-100)
        mask = labels != -100

        # Doğru tahminleri say
        correct = (pred_tokens == labels) & mask
        accuracy = correct.sum().item() / mask.sum().item()

        return accuracy

    def compute_token_accuracy(self, predictions: List[str], references: List[str]) -> float:
        """
        Metin düzeyinde token doğruluğu

        Args:
            predictions: Model çıktıları (metinler)
            references: Gerçek cevaplar (metinler)
        """
        if not self.tokenizer:
            logger.warning("Tokenizer bulunamadı, token accuracy hesaplanamıyor")
            return 0.0

        total_correct = 0
        total_tokens = 0

        for pred, ref in zip(predictions, references):
            pred_tokens = self.tokenizer.tokenize(pred)
            ref_tokens = self.tokenizer.tokenize(ref)

            # En kısa uzunluğa göre karşılaştır
            min_len = min(len(pred_tokens), len(ref_tokens))

            for i in range(min_len):
                if pred_tokens[i] == ref_tokens[i]:
                    total_correct += 1

            total_tokens += len(ref_tokens)

        return total_correct / total_tokens if total_tokens > 0 else 0.0

    def compute_bleu_score(self, predictions: List[str], references: List[str]) -> float:
        """
        Basit BLEU score hesaplama (unigram)

        Args:
            predictions: Model çıktıları
            references: Gerçek cevaplar
        """
        if not predictions or not references:
            return 0.0

        scores = []
        for pred, ref in zip(predictions, references):
            pred_words = set(pred.lower().split())
            ref_words = set(ref.lower().split())

            if len(pred_words) == 0:
                scores.append(0.0)
                continue

            # Kesişim / tahmin sayısı
            overlap = len(pred_words & ref_words)
            score = overlap / len(pred_words)
            scores.append(score)

        return np.mean(scores)

    def evaluate_batch(
        self,
        loss: float,
        predictions: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Bir batch için tüm metrikleri hesapla

        Returns:
            Metrik sözlüğü
        """
        metrics = {
            'loss': loss,
            'perplexity': self.compute_perplexity(loss)
        }

        if predictions is not None and labels is not None:
            metrics['accuracy'] = self.compute_accuracy(predictions, labels)

        return metrics

    def aggregate_metrics(self, metric_list: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Birden fazla batch metriğini birleştir

        Args:
            metric_list: Her batch için metrik sözlükleri

        Returns:
            Ortalama metrikler
        """
        if not metric_list:
            return {}

        # Tüm metrikleri topla
        aggregated = {}
        for key in metric_list[0].keys():
            values = [m[key] for m in metric_list if key in m and not np.isnan(m[key]) and not np.isinf(m[key])]
            if values:
                aggregated[key] = np.mean(values)

        return aggregated


class TurkishTextMetrics:
    """Türkçe'ye özel metin kalite metrikleri"""

    @staticmethod
    def check_turkish_characters(text: str) -> float:
        """
        Türkçe karakterlerin doğru kullanım oranı

        Returns:
            0-1 arası skor (1 = tam doğru)
        """
        turkish_chars = set('çÇğĞıİöÖşŞüÜ')

        # Türkçe karakter içeren kelimeler
        words_with_turkish = sum(1 for word in text.split() if any(c in turkish_chars for c in word))

        if words_with_turkish == 0:
            return 1.0  # Türkçe karakter gerektirmeyen metinler için

        # Basit heuristic: Türkçe karakterlerin varlığı
        return min(1.0, words_with_turkish / max(1, len(text.split()) * 0.3))

    @staticmethod
    def check_sentence_structure(text: str) -> float:
        """
        Cümle yapısı kontrolü

        Returns:
            0-1 arası skor
        """
        sentences = text.split('.')

        # Boş cümleler
        valid_sentences = [s.strip() for s in sentences if s.strip()]

        if not valid_sentences:
            return 0.0

        # Her cümlenin makul uzunlukta olması
        avg_length = np.mean([len(s.split()) for s in valid_sentences])

        # İdeal: 5-20 kelime arası
        if 5 <= avg_length <= 20:
            return 1.0
        elif avg_length < 5:
            return avg_length / 5.0
        else:
            return max(0.5, 20.0 / avg_length)

    @staticmethod
    def check_coherence(text: str) -> float:
        """
        Metin tutarlılığı (basit)

        Returns:
            0-1 arası skor
        """
        # Tekrar eden kelime oranı
        words = text.lower().split()
        if len(words) == 0:
            return 0.0

        unique_words = len(set(words))
        repetition_ratio = unique_words / len(words)

        # İdeal: %60-90 arası unique
        if 0.6 <= repetition_ratio <= 0.9:
            return 1.0
        else:
            return max(0.5, min(repetition_ratio / 0.6, 0.9 / repetition_ratio))


def compute_all_metrics(
    loss: float,
    predictions: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    pred_texts: Optional[List[str]] = None,
    ref_texts: Optional[List[str]] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None
) -> Dict[str, float]:
    """
    Tüm metrikleri tek seferde hesapla

    Args:
        loss: Model loss değeri
        predictions: Logit tahminleri
        labels: Gerçek label'lar
        pred_texts: Üretilen metinler
        ref_texts: Referans metinler
        tokenizer: Tokenizer (opsiyonel)

    Returns:
        Tüm metrikler
    """
    metrics = {}

    # Temel metrikler
    eval_metrics = EvaluationMetrics(tokenizer)
    base_metrics = eval_metrics.evaluate_batch(loss, predictions, labels)
    metrics.update(base_metrics)

    # Metin metrikleri
    if pred_texts and ref_texts:
        metrics['bleu'] = eval_metrics.compute_bleu_score(pred_texts, ref_texts)

        # Türkçe özel metrikler
        turkish_metrics = TurkishTextMetrics()
        turkish_scores = []

        for text in pred_texts:
            score = (
                turkish_metrics.check_turkish_characters(text) * 0.3 +
                turkish_metrics.check_sentence_structure(text) * 0.4 +
                turkish_metrics.check_coherence(text) * 0.3
            )
            turkish_scores.append(score)

        metrics['turkish_quality'] = np.mean(turkish_scores)

    return metrics
