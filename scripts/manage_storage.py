"""
TagAI - Depolama Yönetimi
Google Drive 15GB limiti için otomatik depolama yönetimi
"""

import os
import shutil
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StorageManager:
    """Google Drive depolama yönetimi için yardımcı sınıf"""

    def __init__(self, base_path: str, max_size_gb: float = 15.0, min_free_gb: float = 2.0):
        """
        Args:
            base_path: TagAI proje ana dizini
            max_size_gb: Maksimum kullanılabilir alan (GB)
            min_free_gb: Minimum boş alan (GB)
        """
        self.base_path = Path(base_path)
        self.max_size_bytes = max_size_gb * 1024**3
        self.min_free_bytes = min_free_gb * 1024**3
        self.checkpoint_dir = self.base_path / "models" / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def get_directory_size(self, path: Path) -> int:
        """Dizin boyutunu hesapla (bytes)"""
        total = 0
        try:
            for entry in path.rglob('*'):
                if entry.is_file():
                    total += entry.stat().st_size
        except Exception as e:
            logger.warning(f"Dizin boyutu hesaplanamadı {path}: {e}")
        return total

    def get_total_usage(self) -> Dict[str, int]:
        """Toplam kullanım istatistikleri"""
        usage = {
            'total_bytes': 0,
            'checkpoints_bytes': 0,
            'dataset_bytes': 0,
            'other_bytes': 0
        }

        # Checkpoint boyutu
        if self.checkpoint_dir.exists():
            usage['checkpoints_bytes'] = self.get_directory_size(self.checkpoint_dir)

        # Dataset boyutu
        dataset_dir = self.base_path / "dataset"
        if dataset_dir.exists():
            usage['dataset_bytes'] = self.get_directory_size(dataset_dir)

        # Toplam
        usage['total_bytes'] = self.get_directory_size(self.base_path)
        usage['other_bytes'] = usage['total_bytes'] - usage['checkpoints_bytes'] - usage['dataset_bytes']

        return usage

    def format_size(self, bytes_size: int) -> str:
        """Bytes'ı okunabilir formata çevir"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.2f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.2f} TB"

    def print_usage_report(self):
        """Kullanım raporunu yazdır"""
        usage = self.get_total_usage()

        print("\n" + "="*60)
        print("📊 DEPOLAMA KULLANIM RAPORU")
        print("="*60)
        print(f"Toplam Kullanım: {self.format_size(usage['total_bytes'])} / {self.format_size(self.max_size_bytes)}")
        print(f"  - Checkpoints: {self.format_size(usage['checkpoints_bytes'])}")
        print(f"  - Dataset: {self.format_size(usage['dataset_bytes'])}")
        print(f"  - Diğer: {self.format_size(usage['other_bytes'])}")

        usage_percent = (usage['total_bytes'] / self.max_size_bytes) * 100
        print(f"\nKullanım Oranı: {usage_percent:.1f}%")

        remaining = self.max_size_bytes - usage['total_bytes']
        print(f"Kalan Alan: {self.format_size(remaining)}")

        if remaining < self.min_free_bytes:
            print("\n⚠️  UYARI: Boş alan kritik seviyede!")
        elif usage_percent > 80:
            print("\n⚠️  UYARI: Depolama %80'i geçti!")
        else:
            print("\n✅ Depolama durumu normal")
        print("="*60 + "\n")

    def get_checkpoint_info(self) -> List[Dict]:
        """Checkpoint'lerin bilgilerini al"""
        checkpoints = []

        if not self.checkpoint_dir.exists():
            return checkpoints

        for checkpoint_path in self.checkpoint_dir.iterdir():
            if checkpoint_path.is_dir() and checkpoint_path.name.startswith('checkpoint'):
                try:
                    size = self.get_directory_size(checkpoint_path)
                    mtime = checkpoint_path.stat().st_mtime

                    # Trainer state dosyasından bilgi al
                    trainer_state = checkpoint_path / "trainer_state.json"
                    loss = None
                    if trainer_state.exists():
                        with open(trainer_state, 'r') as f:
                            state = json.load(f)
                            if 'log_history' in state and len(state['log_history']) > 0:
                                # Son loss değerini al
                                for log in reversed(state['log_history']):
                                    if 'loss' in log:
                                        loss = log['loss']
                                        break

                    checkpoints.append({
                        'path': checkpoint_path,
                        'name': checkpoint_path.name,
                        'size': size,
                        'mtime': mtime,
                        'loss': loss,
                        'datetime': datetime.fromtimestamp(mtime)
                    })
                except Exception as e:
                    logger.warning(f"Checkpoint bilgisi okunamadı {checkpoint_path}: {e}")

        return checkpoints

    def cleanup_old_checkpoints(self, keep_last: int = 2, keep_best: bool = True):
        """
        Eski checkpoint'leri temizle

        Args:
            keep_last: Saklanacak son N checkpoint
            keep_best: En iyi checkpoint'i ayrıca sakla
        """
        checkpoints = self.get_checkpoint_info()

        if len(checkpoints) <= keep_last:
            logger.info(f"Checkpoint sayısı ({len(checkpoints)}) limiti aşmıyor. Temizlik gerekmez.")
            return

        # Zamana göre sırala (en yeni en başta)
        checkpoints.sort(key=lambda x: x['mtime'], reverse=True)

        # Saklanacaklar
        to_keep = set()

        # Son N checkpoint'i sakla
        for i in range(min(keep_last, len(checkpoints))):
            to_keep.add(checkpoints[i]['path'])

        # En iyi checkpoint'i sakla (en düşük loss)
        if keep_best:
            best_checkpoint = None
            best_loss = float('inf')

            for cp in checkpoints:
                if cp['loss'] is not None and cp['loss'] < best_loss:
                    best_loss = cp['loss']
                    best_checkpoint = cp

            if best_checkpoint:
                to_keep.add(best_checkpoint['path'])
                logger.info(f"En iyi checkpoint saklanıyor: {best_checkpoint['name']} (loss: {best_loss:.4f})")

        # Silinecekleri temizle
        deleted_size = 0
        deleted_count = 0

        for cp in checkpoints:
            if cp['path'] not in to_keep:
                try:
                    logger.info(f"Siliniyor: {cp['name']} ({self.format_size(cp['size'])})")
                    shutil.rmtree(cp['path'])
                    deleted_size += cp['size']
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Checkpoint silinemedi {cp['name']}: {e}")

        if deleted_count > 0:
            logger.info(f"✅ {deleted_count} checkpoint silindi, {self.format_size(deleted_size)} alan kazanıldı")

    def ensure_space_available(self, required_bytes: int) -> bool:
        """
        Gerekli alanın mevcut olduğundan emin ol, gerekirse temizlik yap

        Args:
            required_bytes: Gerekli alan (bytes)

        Returns:
            Yeterli alan varsa True
        """
        usage = self.get_total_usage()
        available = self.max_size_bytes - usage['total_bytes']

        if available >= required_bytes:
            return True

        logger.warning(f"Yetersiz alan! Gerekli: {self.format_size(required_bytes)}, Mevcut: {self.format_size(available)}")
        logger.info("Otomatik temizlik başlatılıyor...")

        # Checkpoint temizliği
        self.cleanup_old_checkpoints(keep_last=1, keep_best=True)

        # Kontrol et
        usage = self.get_total_usage()
        available = self.max_size_bytes - usage['total_bytes']

        return available >= required_bytes

    def cleanup_temp_files(self):
        """Geçici dosyaları temizle"""
        temp_patterns = ['*.tmp', '*.temp', '*.cache', '__pycache__']
        deleted_size = 0
        deleted_count = 0

        for pattern in temp_patterns:
            for temp_file in self.base_path.rglob(pattern):
                try:
                    size = temp_file.stat().st_size if temp_file.is_file() else self.get_directory_size(temp_file)

                    if temp_file.is_file():
                        temp_file.unlink()
                    else:
                        shutil.rmtree(temp_file)

                    deleted_size += size
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Geçici dosya silinemedi {temp_file}: {e}")

        if deleted_count > 0:
            logger.info(f"✅ {deleted_count} geçici dosya silindi, {self.format_size(deleted_size)} alan kazanıldı")


def main():
    """Test ve CLI kullanımı"""
    import sys

    if len(sys.argv) < 2:
        print("Kullanım: python manage_storage.py <proje_dizini> [komut]")
        print("Komutlar:")
        print("  report - Kullanım raporu")
        print("  cleanup - Checkpoint temizliği")
        print("  temp - Geçici dosya temizliği")
        return

    base_path = sys.argv[1]
    command = sys.argv[2] if len(sys.argv) > 2 else 'report'

    manager = StorageManager(base_path)

    if command == 'report':
        manager.print_usage_report()
    elif command == 'cleanup':
        manager.cleanup_old_checkpoints()
        manager.print_usage_report()
    elif command == 'temp':
        manager.cleanup_temp_files()
        manager.print_usage_report()
    else:
        print(f"Bilinmeyen komut: {command}")


if __name__ == "__main__":
    main()
