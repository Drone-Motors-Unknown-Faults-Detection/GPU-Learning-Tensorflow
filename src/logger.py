import os
import tensorflow as tf
import platform
import subprocess
from datetime import datetime
from typing import Optional, Tuple


def _format_cuda_version_for_report() -> Optional[str]:
    """
    回傳 TensorFlow 所對應的 CUDA 版本字串。
    非 CUDA 建置或無法取得時回傳 None。
    """
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        return None
    try:
        build_info = tf.sysconfig.get_build_info()
        ver = build_info.get('cuda_version', None)
        if ver and str(ver).strip():
            return str(ver).strip()
    except Exception:
        pass
    return None


class TrainingLogger:
    """將訓練過程紀錄並匯出成 txt 檔。enabled=False 時所有方法皆為 no-op。"""

    def __init__(self, enabled: bool = True, device: Optional[str] = None):
        self.enabled = enabled
        self.epoch_records = []
        self.start_time: datetime = datetime.now()
        self.total_time: float = 0.0
        self.host_name: str = platform.node() or "unknown"
        # 完整 Python 版本，例如 3.10.12
        self.python_version: str = platform.python_version()
        self.tf_version: str = tf.__version__

        # 取得裝置資訊，供匯出時寫入
        resolved = device if device is not None else (
            "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"
        )
        self.device_type, self.device_info = self._get_device_info(resolved)
        self.device_name = self.device_info
        # 僅在使用 NVIDIA CUDA GPU 訓練時填入
        self.cuda_version: Optional[str] = None
        if self.device_type == "cuda":
            self.cuda_version = _format_cuda_version_for_report()

    @staticmethod
    def _try_get_cpu_brand_string() -> Optional[str]:
        """
        嘗試取得較具體的 CPU/晶片名稱（macOS 常可拿到如 Apple M4 Pro / Intel...）。
        失敗時回傳 None。
        """
        if platform.system().lower() != "darwin":
            return None
        try:
            out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], stderr=subprocess.DEVNULL)
            brand = out.decode("utf-8", errors="ignore").strip()
            return brand or None
        except Exception:
            return None

    @staticmethod
    def _get_device_info(device: str) -> Tuple[str, str]:
        if "GPU" in device:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    details = tf.config.experimental.get_device_details(gpus[0])
                    name = details.get('device_name', '')
                    if 'compute_capability' in details:
                        return "cuda", name if name else "NVIDIA GPU"
                    if name:
                        return "metal", name
                except Exception:
                    pass
            brand = TrainingLogger._try_get_cpu_brand_string()
            if brand:
                return "metal", brand
            machine = platform.machine() or ""
            if platform.system().lower() == "darwin" and machine.lower() == "arm64":
                return "metal", f"Apple Silicon ({machine})"
            return "gpu", "GPU"

        return "cpu", "CPU"

    def start(self):
        """記錄訓練開始時間，應在訓練迴圈前呼叫。"""
        if not self.enabled:
            return
        self.start_time = datetime.now()

    def log_epoch(self, epoch: int, total_epochs: int, loss: float, train_acc: float, test_acc: float, elapsed: float):
        """記錄單一 epoch 的結果，應在每個 epoch 結束後呼叫。"""
        if not self.enabled:
            return
        self.epoch_records.append({
            "epoch": epoch,
            "total_epochs": total_epochs,
            "loss": loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "elapsed": elapsed,
        })

    def finish(self, total_time: float):
        """記錄總訓練時間，應在訓練迴圈結束後呼叫。"""
        if not self.enabled:
            return
        self.total_time = total_time

    def export(self, title: str, output_dir: str = "."):
        """將完整訓練紀錄寫入 txt，檔名包含訓練開始時間。"""
        if not self.enabled:
            return
        os.makedirs(output_dir, exist_ok=True)
        filename = self.start_time.strftime("training_log_%Y%m%d_%H%M%S.txt")
        filepath = os.path.join(output_dir, filename)

        lines = []
        lines.append("=" * 50)
        lines.append(title)
        lines.append("=" * 50)
        lines.append(f"電腦名稱         : {self.host_name}")
        lines.append(f"Python 版本      : {self.python_version}")
        lines.append(f"TensorFlow 版本  : {self.tf_version}")
        lines.append(f"訓練裝置類型     : {self.device_type.upper()}")
        lines.append(f"裝置名稱         : {self.device_info}")
        if self.device_type == "cuda":
            cuda_line = self.cuda_version if self.cuda_version else "未知（TensorFlow 未回報 CUDA 版本）"
            lines.append(f"CUDA 版本        : {cuda_line}")
        lines.append(f"訓練開始時間     : {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"總訓練時間       : {self.total_time:.2f} 秒")
        lines.append("")
        lines.append("-" * 50)
        lines.append(f"{'Epoch':<8} {'Loss':<10} {'Train Acc':<12} {'Test Acc':<12} {'累計時間'}")
        lines.append("-" * 50)
        for r in self.epoch_records:
            lines.append(
                f"{r['epoch']}/{r['total_epochs']:<5} "
                f"{r['loss']:<10.4f} "
                f"{r['train_acc']:<12.2f} "
                f"{r['test_acc']:<12.2f} "
                f"{r['elapsed']:.2f}s"
            )
        lines.append("-" * 50)
        lines.append("")
        lines.append("訓練完成")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"訓練紀錄已儲存至 {filepath}")
