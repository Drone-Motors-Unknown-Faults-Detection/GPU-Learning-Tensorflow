import platform
from dataclasses import dataclass
from typing import Optional, Tuple

import tensorflow as tf


@dataclass(frozen=True)
class MacChipInfo:
    is_macos: bool
    is_apple_silicon: bool
    machine: str
    processor: str


def get_mac_chip_info() -> MacChipInfo:
    is_macos = platform.system().lower() == "darwin"
    machine = platform.machine() or ""
    processor = platform.processor() or ""
    # Apple Silicon Macs report arm64.
    is_apple_silicon = bool(is_macos and machine.lower() == "arm64")
    return MacChipInfo(
        is_macos=is_macos,
        is_apple_silicon=is_apple_silicon,
        machine=machine,
        processor=processor,
    )


def get_best_tf_device() -> str:
    """
    選擇最適合的 TensorFlow device 字串：
    - NVIDIA CUDA GPU 或 Apple Metal GPU：/GPU:0
    - 其他：/CPU:0
    TensorFlow 會自動將運算分配至可用的 GPU，此函式主要用於顯示與紀錄。
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        return "/GPU:0"
    return "/CPU:0"


def get_device_display_info(device: str) -> Tuple[str, Optional[str]]:
    """
    回傳 (device_type, device_detail) 用於顯示/紀錄。
    device_type: 'cuda'（NVIDIA）、'metal'（Apple）、'gpu'（其他 GPU）、'cpu'
    """
    if "GPU" in device:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                details = tf.config.experimental.get_device_details(gpus[0])
                name = details.get('device_name', '')
                # NVIDIA GPU 具有 compute_capability 欄位
                if 'compute_capability' in details:
                    return "cuda", name if name else "NVIDIA GPU"
                if name:
                    chip = get_mac_chip_info()
                    if chip.is_apple_silicon:
                        return "metal", name
                    return "gpu", name
            except Exception:
                pass
        chip = get_mac_chip_info()
        if chip.is_apple_silicon:
            return "metal", f"Apple Silicon ({chip.machine})"
        return "gpu", "GPU"

    return "cpu", "CPU"
