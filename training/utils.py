"""
TagAI - Eğitim Yardımcı Fonksiyonları
"""

import os
import json
import torch
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict:
    """Konfigürasyon dosyasını yükle"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"✅ Config yüklendi: {config_path}")
        return config
    except Exception as e:
        logger.error(f"❌ Config yüklenemedi: {e}")
        raise


def load_json_dataset(file_path: str) -> List[Dict]:
    """JSON veri dosyasını yükle"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"✅ Veri yüklendi: {file_path} ({len(data)} örnek)")
        return data
    except Exception as e:
        logger.error(f"❌ Veri yüklenemedi {file_path}: {e}")
        return []


def combine_datasets(data_dir: str = "dataset/raw") -> List[Dict]:
    """Tüm JSON dosyalarını birleştir"""
    data_path = Path(data_dir)
    all_data = []

    if not data_path.exists():
        logger.warning(f"⚠️  Veri dizini bulunamadı: {data_dir}")
        return all_data

    for json_file in data_path.glob("*.json"):
        data = load_json_dataset(str(json_file))
        all_data.extend(data)

    logger.info(f"✅ Toplam {len(all_data)} örnek birleştirildi")
    return all_data


def format_prompt(instruction: str, response: str = "") -> str:
    """
    Alpaca stili prompt formatı

    Args:
        instruction: Kullanıcı talimatı
        response: Model yanıtı (training için)
    """
    if response:
        return f"""### Talimat:
{instruction}

### Yanıt:
{response}"""
    else:
        return f"""### Talimat:
{instruction}

### Yanıt:
"""


def format_chat_prompt(messages: List[Dict[str, str]]) -> str:
    """
    Chat formatında prompt oluştur

    Args:
        messages: [{"role": "user"/"assistant", "content": "..."}]
    """
    formatted = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "user":
            formatted.append(f"Kullanıcı: {content}")
        elif role == "assistant":
            formatted.append(f"Asistan: {content}")

    return "\n\n".join(formatted)


def preprocess_function(
    examples: Dict,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048
) -> Dict:
    """
    Veri setini tokenize et

    Args:
        examples: Batch of examples
        tokenizer: Tokenizer
        max_length: Maksimum sequence uzunluğu
    """
    # Prompt formatla
    prompts = []
    for instruction, response in zip(examples['instruction'], examples['response']):
        prompt = format_prompt(instruction, response)
        prompts.append(prompt)

    # Tokenize
    model_inputs = tokenizer(
        prompts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors=None
    )

    # Labels = inputs (language modeling için)
    model_inputs["labels"] = model_inputs["input_ids"].copy()

    return model_inputs


def prepare_dataset(
    data: List[Dict],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
    train_split: float = 0.95
) -> DatasetDict:
    """
    Veri setini eğitim için hazırla

    Args:
        data: Ham veri listesi
        tokenizer: Tokenizer
        max_length: Maksimum sequence uzunluğu
        train_split: Train/validation oranı

    Returns:
        DatasetDict with train/validation splits
    """
    # Dataset oluştur
    dataset = Dataset.from_list(data)

    # Train/validation split
    split_dataset = dataset.train_test_split(
        test_size=1.0 - train_split,
        seed=42
    )

    # Tokenize
    tokenized_datasets = split_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=split_dataset["train"].column_names,
        desc="Tokenizing dataset"
    )

    logger.info(f"✅ Dataset hazırlandı:")
    logger.info(f"   Train: {len(tokenized_datasets['train'])} örnek")
    logger.info(f"   Validation: {len(tokenized_datasets['test'])} örnek")

    return tokenized_datasets


def print_gpu_memory():
    """GPU bellek kullanımını yazdır"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3

        print(f"\n🔧 GPU Bellek Kullanımı:")
        print(f"   Allocated: {allocated:.2f} GB")
        print(f"   Reserved: {reserved:.2f} GB")
        print(f"   Total: {total:.2f} GB")
        print(f"   Free: {total - reserved:.2f} GB\n")


def count_trainable_parameters(model) -> tuple:
    """
    Model parametrelerini say

    Returns:
        (trainable_params, total_params, trainable_percentage)
    """
    trainable_params = 0
    all_params = 0

    for param in model.parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    trainable_pct = 100 * trainable_params / all_params if all_params > 0 else 0

    return trainable_params, all_params, trainable_pct


def print_model_info(model):
    """Model bilgilerini yazdır"""
    trainable, total, pct = count_trainable_parameters(model)

    print("\n" + "="*60)
    print("🤖 MODEL BİLGİLERİ")
    print("="*60)
    print(f"Toplam Parametreler: {total:,}")
    print(f"Eğitilebilir Parametreler: {trainable:,}")
    print(f"Eğitilebilir Oran: {pct:.2f}%")
    print("="*60 + "\n")


def save_training_args(args: Any, output_dir: str):
    """Eğitim argümanlarını kaydet"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    args_dict = args.to_dict() if hasattr(args, 'to_dict') else vars(args)

    with open(output_path / "training_args.json", 'w', encoding='utf-8') as f:
        json.dump(args_dict, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ Eğitim argümanları kaydedildi: {output_path / 'training_args.json'}")


def load_checkpoint_info(checkpoint_dir: str) -> Optional[Dict]:
    """Checkpoint bilgilerini yükle"""
    checkpoint_path = Path(checkpoint_dir)

    if not checkpoint_path.exists():
        return None

    trainer_state_file = checkpoint_path / "trainer_state.json"

    if trainer_state_file.exists():
        with open(trainer_state_file, 'r') as f:
            return json.load(f)

    return None


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """En son checkpoint'i bul"""
    checkpoint_path = Path(checkpoint_dir)

    if not checkpoint_path.exists():
        return None

    checkpoints = list(checkpoint_path.glob("checkpoint-*"))

    if not checkpoints:
        return None

    # Checkpoint numarasına göre sırala
    checkpoints.sort(key=lambda x: int(x.name.split("-")[1]))

    latest = str(checkpoints[-1])
    logger.info(f"📁 En son checkpoint bulundu: {latest}")

    return latest


def clean_memory():
    """Belleği temizle"""
    import gc
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def setup_logging(log_dir: str = "logs"):
    """Logging kurulumu"""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    log_file = log_path / f"training_{os.getpid()}.log"

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.info(f"✅ Logging başlatıldı: {log_file}")


def validate_config(config: Dict) -> bool:
    """Konfigürasyonu doğrula"""
    required_keys = ['model', 'training', 'dataset', 'storage']

    for key in required_keys:
        if key not in config:
            logger.error(f"❌ Eksik config anahtarı: {key}")
            return False

    # Model config kontrolü
    if 'base_model' not in config['model']:
        logger.error("❌ base_model belirtilmemiş")
        return False

    logger.info("✅ Config doğrulandı")
    return True


def get_model_size_mb(model) -> float:
    """Model boyutunu MB cinsinden hesapla"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())

    total_size_mb = (param_size + buffer_size) / 1024**2
    return total_size_mb
