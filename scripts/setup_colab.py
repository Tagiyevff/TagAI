"""
TagAI - Google Colab Ortam Kurulum Scripti
GPU tipini otomatik tespit eder ve ayarları optimize eder
"""

import os
import sys
import subprocess
import yaml
import torch
import psutil
from pathlib import Path
from typing import Dict, Optional


class ColabSetup:
    """Google Colab ortamını hazırlar ve optimize eder"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        self.gpu_type = None
        self.drive_path = None

    def load_config(self) -> Dict:
        """Konfigürasyon dosyasını yükle"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"⚠️  Config yüklenemedi: {e}")
            return {}

    def detect_gpu(self) -> Optional[str]:
        """GPU tipini tespit et"""
        if not torch.cuda.is_available():
            print("❌ GPU bulunamadı! CPU modunda çalışılacak.")
            return None

        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU bulundu: {gpu_name}")

        # GPU tipini belirle
        if "T4" in gpu_name:
            gpu_type = "T4"
        elif "L4" in gpu_name:
            gpu_type = "L4"
        elif "A100" in gpu_name:
            gpu_type = "A100"
        elif "V100" in gpu_name:
            gpu_type = "V100"
        else:
            print(f"⚠️  Bilinmeyen GPU tipi: {gpu_name}")
            gpu_type = "T4"  # Varsayılan

        self.gpu_type = gpu_type
        return gpu_type

    def get_gpu_config(self) -> Dict:
        """GPU tipine göre optimal konfigürasyon"""
        if not self.gpu_type:
            return {
                'batch_size': 1,
                'gradient_accumulation_steps': 8,
                'fp16': False,
                'bf16': False,
                'max_seq_length': 1024
            }

        gpu_configs = self.config.get('colab', {}).get('gpu_configs', {})
        return gpu_configs.get(self.gpu_type, gpu_configs.get('T4', {}))

    def print_system_info(self):
        """Sistem bilgilerini yazdır"""
        print("\n" + "="*60)
        print("🖥️  SİSTEM BİLGİLERİ")
        print("="*60)

        # CPU
        print(f"CPU Çekirdek Sayısı: {psutil.cpu_count()}")
        print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")

        # GPU
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Belleği: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
            print(f"CUDA Versiyonu: {torch.version.cuda}")
        else:
            print("GPU: Yok")

        # Python
        print(f"Python Versiyonu: {sys.version.split()[0]}")
        print(f"PyTorch Versiyonu: {torch.__version__}")

        print("="*60 + "\n")

    def mount_drive(self) -> bool:
        """Google Drive'ı bağla"""
        try:
            # Colab içinde mi kontrol et
            try:
                from google.colab import drive
                in_colab = True
            except ImportError:
                print("⚠️  Google Colab ortamı değil, Drive mount atlanıyor")
                return False

            if in_colab:
                print("📁 Google Drive bağlanıyor...")
                drive.mount('/content/drive')

                # TagAI dizinini oluştur
                drive_base = Path('/content/drive/MyDrive/TagAI')
                drive_base.mkdir(parents=True, exist_ok=True)

                self.drive_path = str(drive_base)
                print(f"✅ Drive bağlandı: {self.drive_path}")

                # Sembollk link oluştur (kolay erişim için)
                if not Path('TagAI_Drive').exists():
                    os.symlink(self.drive_path, 'TagAI_Drive')

                return True

        except Exception as e:
            print(f"❌ Drive bağlanamadı: {e}")
            return False

    def install_dependencies(self):
        """Gerekli kütüphaneleri yükle"""
        print("\n📦 Bağımlılıklar yükleniyor...")

        requirements = [
            "torch>=2.1.0",
            "transformers>=4.36.0",
            "datasets>=2.15.0",
            "accelerate>=0.25.0",
            "peft>=0.7.0",
            "bitsandbytes>=0.41.0",
            "trl>=0.7.0",
            "pyyaml>=6.0",
            "tqdm>=4.66.0",
            "sentencepiece>=0.1.99",
        ]

        # Unsloth (opsiyonel ama önerilen)
        try:
            print("  - Unsloth kuruluyor (optimizasyon için)...")
            subprocess.run(
                ["pip", "install", "unsloth", "-q"],
                check=False,
                capture_output=True
            )
            print("  ✅ Unsloth kuruldu")
        except:
            print("  ⚠️  Unsloth kurulamadı (opsiyonel)")

        print("\n✅ Tüm bağımlılıklar yüklendi")

    def setup_directories(self):
        """Proje dizinlerini oluştur"""
        print("\n📁 Proje dizinleri oluşturuluyor...")

        dirs = [
            'dataset/raw',
            'dataset/processed',
            'dataset/synthetic',
            'models/checkpoints',
            'models/output',
            'logs',
            'training/config',
            'evaluation',
            'scripts',
        ]

        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        print("✅ Dizinler oluşturuldu")

    def create_training_config(self):
        """GPU tipine göre eğitim konfigürasyonu oluştur"""
        gpu_config = self.get_gpu_config()

        training_config = {
            'model': self.config.get('model', {}),
            'training': {
                **self.config.get('training', {}),
                **gpu_config  # GPU-specific overrides
            },
            'dataset': self.config.get('dataset', {}),
            'storage': self.config.get('storage', {}),
        }

        # Config dosyasını kaydet
        config_path = Path('training/config/training_config.yaml')
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(training_config, f, allow_unicode=True, default_flow_style=False)

        print(f"\n✅ Eğitim konfigürasyonu oluşturuldu: {config_path}")
        print(f"   GPU Tipi: {self.gpu_type or 'CPU'}")
        print(f"   Batch Size: {gpu_config.get('batch_size', 'N/A')}")
        print(f"   Gradient Accumulation: {gpu_config.get('gradient_accumulation_steps', 'N/A')}")
        print(f"   Precision: {'bf16' if gpu_config.get('bf16') else 'fp16' if gpu_config.get('fp16') else 'fp32'}")

    def clear_memory(self):
        """Gereksiz belleği temizle"""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("🧹 Bellek temizlendi")

    def run_setup(self) -> Dict:
        """Tam kurulum prosedürünü çalıştır"""
        print("\n" + "="*60)
        print("🚀 TagAI - Colab Ortamı Hazırlanıyor")
        print("="*60 + "\n")

        # 1. Sistem bilgileri
        self.print_system_info()

        # 2. GPU tespit
        self.detect_gpu()

        # 3. Drive bağla
        drive_mounted = self.mount_drive()

        # 4. Dizinleri oluştur
        self.setup_directories()

        # 5. Eğitim config oluştur
        self.create_training_config()

        # 6. Belleği temizle
        self.clear_memory()

        print("\n" + "="*60)
        print("✅ Kurulum tamamlandı!")
        print("="*60 + "\n")

        return {
            'gpu_type': self.gpu_type,
            'drive_path': self.drive_path,
            'drive_mounted': drive_mounted,
            'config': self.get_gpu_config()
        }


def setup_colab(config_path: str = "config.yaml") -> Dict:
    """
    Colab ortamını hazırla (main fonksiyon)

    Returns:
        Setup bilgileri (GPU tipi, drive path vb.)
    """
    setup = ColabSetup(config_path)
    return setup.run_setup()


if __name__ == "__main__":
    # Doğrudan çalıştırıldığında
    setup_colab()
