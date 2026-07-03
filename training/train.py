"""
TagAI - Ana Eğitim Scripti
QLoRA/PEFT ile verimli model eğitimi
"""

import os
import sys
import torch
import gc
from pathlib import Path
from typing import Optional, Dict
import logging

# Transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)

# PEFT
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)

# Local imports
from utils import (
    load_config,
    combine_datasets,
    prepare_dataset,
    print_gpu_memory,
    print_model_info,
    save_training_args,
    find_latest_checkpoint,
    clean_memory,
    setup_logging,
    validate_config
)

# Storage management
sys.path.append(str(Path(__file__).parent.parent))
from scripts.manage_storage import StorageManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TagAITrainer:
    """TagAI eğitim sınıfı"""

    def __init__(self, config_path: str = "training/config/training_config.yaml"):
        """
        Args:
            config_path: Eğitim konfigürasyon dosyası
        """
        self.config = load_config(config_path)

        if not validate_config(self.config):
            raise ValueError("Config doğrulaması başarısız!")

        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.trainer = None

        # Storage manager
        storage_config = self.config.get('storage', {})
        self.storage_manager = StorageManager(
            base_path=".",
            max_size_gb=storage_config.get('max_size_gb', 15),
            min_free_gb=storage_config.get('min_free_space_gb', 2)
        )

        setup_logging()

    def load_model_and_tokenizer(self):
        """Model ve tokenizer'ı yükle"""
        logger.info("🤖 Model yükleniyor...")

        model_config = self.config['model']
        training_config = self.config['training']

        # Tokenizer
        base_model = model_config['base_model']
        logger.info(f"Base model: {base_model}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True
        )

        # Padding token ayarla
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if training_config.get('bf16', False) else torch.float16
        )

        # Model yükle
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if training_config.get('bf16', False) else torch.float16
        )

        # Gradient checkpointing
        if training_config.get('gradient_checkpointing', True):
            self.model.gradient_checkpointing_enable()

        # PEFT hazırlığı
        self.model = prepare_model_for_kbit_training(self.model)

        # LoRA config
        lora_config = LoraConfig(
            r=training_config.get('lora_r', 16),
            lora_alpha=training_config.get('lora_alpha', 32),
            target_modules=training_config.get('target_modules', [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),
            lora_dropout=training_config.get('lora_dropout', 0.05),
            bias="none",
            task_type="CAUSAL_LM"
        )

        # LoRA uygula
        self.model = get_peft_model(self.model, lora_config)

        logger.info("✅ Model ve tokenizer yüklendi")
        print_model_info(self.model)
        print_gpu_memory()

    def prepare_data(self):
        """Veri setini hazırla"""
        logger.info("📊 Veri seti hazırlanıyor...")

        # Veri setini yükle
        dataset_config = self.config['dataset']
        data = combine_datasets("dataset/raw")

        if len(data) == 0:
            raise ValueError("❌ Veri seti boş! Lütfen önce veri oluşturun.")

        logger.info(f"✅ {len(data)} örnek yüklendi")

        # Dataset hazırla
        model_config = self.config['model']
        max_seq_length = model_config.get('max_seq_length', 2048)
        train_split = dataset_config.get('train_split', 0.95)

        self.dataset = prepare_dataset(
            data=data,
            tokenizer=self.tokenizer,
            max_length=max_seq_length,
            train_split=train_split
        )

        logger.info("✅ Veri seti hazır")

    def get_training_arguments(self) -> TrainingArguments:
        """Training arguments oluştur"""
        training_config = self.config['training']
        storage_config = self.config['storage']

        output_dir = "models/output"
        checkpoint_dir = "models/checkpoints"

        # Dizinleri oluştur
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        args = TrainingArguments(
            # Output
            output_dir=output_dir,

            # Training
            num_train_epochs=training_config.get('num_train_epochs', 3),
            per_device_train_batch_size=training_config.get('per_device_train_batch_size', 2),
            gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 4),
            learning_rate=training_config.get('learning_rate', 2e-4),
            weight_decay=training_config.get('weight_decay', 0.01),
            max_grad_norm=training_config.get('max_grad_norm', 1.0),
            warmup_steps=training_config.get('warmup_steps', 100),

            # Precision
            bf16=training_config.get('bf16', True),
            fp16=training_config.get('fp16', False),

            # Optimizer
            optim=training_config.get('optim', 'paged_adamw_8bit'),
            lr_scheduler_type=training_config.get('lr_scheduler_type', 'cosine'),

            # Logging
            logging_steps=training_config.get('logging_steps', 10),
            logging_dir="logs",

            # Evaluation
            eval_strategy="steps",
            eval_steps=training_config.get('eval_steps', 500),
            per_device_eval_batch_size=training_config.get('per_device_train_batch_size', 2),

            # Saving
            save_strategy="steps",
            save_steps=training_config.get('save_steps', 500),
            save_total_limit=storage_config.get('checkpoint_limit', 2),
            load_best_model_at_end=training_config.get('load_best_model_at_end', True),
            metric_for_best_model=training_config.get('metric_for_best_model', 'loss'),

            # Performance
            gradient_checkpointing=training_config.get('gradient_checkpointing', True),
            dataloader_num_workers=0,  # Colab için 0

            # Misc
            report_to="none",  # WandB kapalı (isteğe bağlı açılabilir)
            remove_unused_columns=False,
            push_to_hub=False,
        )

        return args

    def create_trainer(self) -> Trainer:
        """Trainer oluştur"""
        logger.info("🏋️ Trainer oluşturuluyor...")

        training_args = self.get_training_arguments()

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM için False
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
            data_collator=data_collator,
        )

        logger.info("✅ Trainer hazır")
        save_training_args(training_args, training_args.output_dir)

        return trainer

    def train(self, resume_from_checkpoint: Optional[str] = None):
        """
        Eğitimi başlat

        Args:
            resume_from_checkpoint: Devam edilecek checkpoint dizini
        """
        logger.info("🚀 Eğitim başlatılıyor...")

        # Depolama durumunu kontrol et
        self.storage_manager.print_usage_report()

        # Checkpoint temizliği
        if self.config['storage'].get('auto_cleanup', True):
            logger.info("🧹 Eski checkpoint'ler temizleniyor...")
            self.storage_manager.cleanup_old_checkpoints(
                keep_last=self.config['storage'].get('checkpoint_limit', 2),
                keep_best=self.config['storage'].get('keep_best', True)
            )

        # Checkpoint'ten devam mı?
        if resume_from_checkpoint is None:
            resume_from_checkpoint = find_latest_checkpoint("models/checkpoints")

        if resume_from_checkpoint:
            logger.info(f"📁 Checkpoint'ten devam ediliyor: {resume_from_checkpoint}")

        # Belleği temizle
        clean_memory()
        print_gpu_memory()

        try:
            # Eğitim
            logger.info("🏋️ Eğitim başlıyor...")
            self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)

            logger.info("✅ Eğitim tamamlandı!")

            # Final evaluation
            logger.info("📊 Final değerlendirme yapılıyor...")
            eval_results = self.trainer.evaluate()

            print("\n" + "="*60)
            print("📊 FİNAL DEĞERLENDİRME SONUÇLARI")
            print("="*60)
            for key, value in eval_results.items():
                print(f"{key}: {value:.4f}")
            print("="*60 + "\n")

        except KeyboardInterrupt:
            logger.warning("⚠️ Eğitim kullanıcı tarafından durduruldu")
            print("\n⚠️ Eğitim durduruldu. Son checkpoint kaydedildi.")

        except Exception as e:
            logger.error(f"❌ Eğitim hatası: {e}")
            raise

        finally:
            # Temizlik
            clean_memory()

            # Final depolama raporu
            print("\n📊 Final Depolama Durumu:")
            self.storage_manager.print_usage_report()

    def save_model(self, output_path: str = "models/TagAI-final"):
        """
        Eğitilmiş modeli kaydet

        Args:
            output_path: Kayıt dizini
        """
        logger.info(f"💾 Model kaydediliyor: {output_path}")

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Depolama kontrolü
        required_space = 5 * 1024**3  # ~5GB tahmin
        if not self.storage_manager.ensure_space_available(required_space):
            logger.warning("⚠️ Yetersiz depolama! Eski checkpointler temizleniyor...")
            self.storage_manager.cleanup_old_checkpoints(keep_last=0, keep_best=False)

        # Model ve tokenizer kaydet
        self.trainer.save_model(output_path)
        self.tokenizer.save_pretrained(output_path)

        logger.info(f"✅ Model kaydedildi: {output_path}")

        # Model kartı oluştur
        self._create_model_card(output_path)

    def _create_model_card(self, output_path: str):
        """Model card (README) oluştur"""
        model_card = f"""# TagAI - Türkçe Dil Modeli

Bu model TagAI projesi ile eğitilmiştir.

## Model Detayları

- **Base Model**: {self.config['model']['base_model']}
- **Eğitim Yöntemi**: QLoRA + PEFT
- **Dil**: Türkçe
- **Lisans**: MIT

## Kullanım

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Base model ve tokenizer
base_model = "{self.config['model']['base_model']}"
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Model yükle
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    torch_dtype=torch.float16
)

# LoRA adaptörlerini yükle
model = PeftModel.from_pretrained(model, "{output_path}")

# Inference
prompt = "Türkçe dilinde merhaba nasıl denir?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Eğitim Bilgileri

- **Eğitim Verisi**: Özel oluşturulmuş Türkçe veri seti
- **Kategoriler**: Kod yazma, mantık yürütme, genel bilgi, konuşma
- **Eğitim Süresi**: {self.config['training'].get('num_train_epochs', 'N/A')} epoch

## Sınırlamalar

- Model eğitim ve araştırma amaçlıdır
- Üretim kullanımı için kapsamlı test gerektirir
- Türkçe dışındaki dillerde performans sınırlı olabilir

## Katkıda Bulunma

TagAI açık kaynak bir projedir. Katkılarınızı bekliyoruz!

---

**Not**: Bu model TagAI projesi kapsamında üretilmiştir.
"""

        readme_path = Path(output_path) / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(model_card)

        logger.info(f"✅ Model card oluşturuldu: {readme_path}")

    def run(self, resume_from_checkpoint: Optional[str] = None):
        """
        Tam eğitim pipeline'ı çalıştır

        Args:
            resume_from_checkpoint: Devam edilecek checkpoint
        """
        print("\n" + "="*60)
        print("🚀 TagAI EĞİTİM BAŞLIYOR")
        print("="*60 + "\n")

        try:
            # 1. Model ve tokenizer yükle
            if self.model is None:
                self.load_model_and_tokenizer()

            # 2. Veri hazırla
            if self.dataset is None:
                self.prepare_data()

            # 3. Trainer oluştur
            if self.trainer is None:
                self.trainer = self.create_trainer()

            # 4. Eğitimi başlat
            self.train(resume_from_checkpoint)

            # 5. Modeli kaydet
            self.save_model()

            print("\n" + "="*60)
            print("✅ EĞİTİM TAMAMLANDI!")
            print("="*60 + "\n")

            print("📁 Model konumu: models/TagAI-final")
            print("📊 Loglar: logs/")
            print("💾 Checkpoints: models/checkpoints/")

        except Exception as e:
            logger.error(f"❌ Eğitim başarısız: {e}")
            print(f"\n❌ Hata: {e}")
            raise


def main():
    """Ana fonksiyon"""
    import argparse

    parser = argparse.ArgumentParser(description="TagAI Eğitim Scripti")
    parser.add_argument(
        "--config",
        type=str,
        default="training/config/training_config.yaml",
        help="Config dosyası yolu"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Devam edilecek checkpoint dizini"
    )

    args = parser.parse_args()

    # Eğitim başlat
    trainer = TagAITrainer(config_path=args.config)
    trainer.run(resume_from_checkpoint=args.resume)


if __name__ == "__main__":
    main()
