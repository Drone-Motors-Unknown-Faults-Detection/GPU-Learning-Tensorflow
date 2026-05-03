# 環境診斷報告

**日期：** 2026-05-03
**主機：** aero01
**專案：** GPU-Learning-Tensorflow

---

## 一、錯誤現象

```
(venv) albert@aero01:~/GPU-Learning-Tensorflow$ python3 src/mnist.py
Illegal instruction (core dumped)
```

程式在啟動時直接 crash，連任何輸出都沒有。

---

## 二、環境概況

| 項目 | 數值 |
|------|------|
| OS | Ubuntu Linux 5.15.0-176-generic |
| Python | 3.10.12 |
| TensorFlow | 2.17.0 |
| NumPy | 1.23.5 |
| Keras | 3.12.1 |
| RAM | 62 GiB（可用 60 GiB） |
| GPU | NVIDIA L40S（46 GiB VRAM） |
| GPU Driver | 590.48.01（支援 CUDA 最高 13.1） |
| GPU 計算能力 | Compute Capability 8.9 |
| CPU | QEMU Virtual CPU version 2.5+ |

---

## 三、問題分析

### 問題 1（Critical）— CPU 缺少 AVX 指令集

**這是造成 crash 的直接原因。**

```
CPU flags: sse, sse2, sse3, ssse3, sse4_1, sse4_2
缺少：avx, avx2, avx512
```

TensorFlow 2.6 以後的官方預編譯版本要求 CPU 支援 **AVX 指令集**。目前系統是 QEMU 虛擬機，且虛擬機設定沒有將宿主機的 AVX 指令集暴露給 guest，導致 `import tensorflow` 時執行到 AVX 指令，CPU 無法識別，產生 `SIGILL`（Illegal Instruction）並 core dump。

**注意：** 這個 crash 發生在程式進入 `main()` 之前，純粹是 import 階段就死掉，與程式碼邏輯無關。

---

### 問題 2（Critical）— CUDA Runtime 未安裝

GPU driver 已存在，但 CUDA toolkit 未安裝：

```
✅ 已安裝：libcuda.so（GPU Driver，支援 CUDA 13.1）
❌ 缺少：libcudart.so（CUDA Runtime）
❌ 缺少：libcublas.so（cuBLAS）
❌ 缺少：libcudnn.so（cuDNN）
```

TensorFlow 2.17.0 需要 **CUDA 12.3 + cuDNN 8.9** 才能使用 GPU。即使問題 1 解決，若 CUDA Runtime 未安裝，TF 仍只能跑 CPU（且速度會比應有的慢），或產生 GPU 找不到的警告。

---

## 四、修復建議

### 修復問題 1：暴露 AVX 指令集

**方法 A（推薦）：** 請有 hypervisor 管理權的人修改 VM 設定，讓 VM 使用宿主機 CPU 型號：

```xml
<!-- libvirt XML -->
<cpu mode='host-passthrough'/>
```

或 QEMU 啟動參數：

```bash
-cpu host
```

**方法 B（無需動 VM）：** 降版至不要求 AVX 的 TF：

```bash
pip install tensorflow==2.3.0
```

但 2.3.0 過舊（2020 年），許多新 API 不支援，不建議長期使用。

---

### 修復問題 2：安裝 CUDA 12.3 Toolkit

```bash
# 安裝 CUDA 12.3（需 sudo）
wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda_12.3.0_545.23.06_linux.run
sudo sh cuda_12.3.0_545.23.06_linux.run --toolkit --silent --override

# 加入環境變數
echo 'export PATH=/usr/local/cuda-12.3/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

安裝完後確認 TF 能偵測到 GPU：

```bash
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

## 五、問題優先順序

```
[1] 修問題 1（AVX）→ 讓 TF 能正常 import
[2] 修問題 2（CUDA）→ 讓 GPU 可被 TF 使用
```

問題 1 不解決，其他什麼都跑不起來。
