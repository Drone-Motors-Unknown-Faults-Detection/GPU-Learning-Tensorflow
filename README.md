# CNN 模型運行 GPU 測試程式

用 CNN / ResNet 模型訓練影像分類，包含 MNIST 手寫數字、CIFAR-10 與 CIFAR-100 彩色圖片，並說明如何讓訓練跑在 GPU 上。

各模型詳細技術說明請見 [`docs/`](docs/) 資料夾。

---

## MNIST（手寫數字辨識）

**腳本**：`src/mnist.py`

### 資料集

- 訓練集：60,000 張 28×28 灰階圖
- 測試集：10,000 張
- 類別：數字 0 ~ 9，共 10 類

### 模型架構

兩層卷積 + 兩層全連接的簡單 CNN：

```
輸入 (28×28×1)
→ Conv2D(32, 5×5) + ReLU + MaxPool  →  14×14×32
→ Conv2D(64, 5×5) + ReLU + MaxPool  →  7×7×64
→ Flatten → Dense(3136→128) + ReLU + Dropout(0.5)
→ Dense(128→10)  →  10 個類別
```

### 超參數

| 項目 | 值 |
|------|----|
| Batch Size | 128 |
| Learning Rate | 0.001 |
| Epochs | 50 |
| Optimizer | Adam |

### 執行

```bash
python src/mnist.py
```

訓練紀錄輸出至 `logs/MNIST/`

---

## CIFAR-10（彩色圖片分類）

**腳本**：`src/cifar10.py`

### 資料集

- 訓練集：50,000 張 32×32 彩色圖（RGB 3 通道）
- 測試集：10,000 張
- 類別：飛機、汽車、鳥、貓、鹿、狗、青蛙、馬、船、卡車，共 10 類

### 模型架構

簡化版 ResNet，三組殘差層逐步縮小特徵圖：

```
輸入 (32×32×3)
→ Conv2D(64, 3×3) + BN + ReLU
→ Layer1: 2× ResidualBlock(64→64)   →  32×32×64
→ Layer2: 2× ResidualBlock(64→128)  →  16×16×128
→ Layer3: 2× ResidualBlock(128→256) →  8×8×256
→ GlobalAveragePooling → Flatten
→ Dense(256→10)  →  10 個類別
```

ResidualBlock 的 shortcut 連接讓梯度可以跳層傳遞，解決深層網路難以訓練的問題。

### 超參數

| 項目 | 值 |
|------|----|
| Batch Size | 256 |
| Learning Rate | 0.001（CosineDecay 衰減） |
| Epochs | 20 |
| Optimizer | Adam |
| LR Scheduler | CosineDecay |
| Weight Decay | 1e-4（L2 正則化） |

訓練集使用隨機裁切（RandomCrop）、水平翻轉（RandomHorizontalFlip）與色彩抖動（ColorJitter）資料增強。

### 執行

```bash
python src/cifar10.py
```

訓練紀錄輸出至 `logs/CIFAR10/`

---

## CIFAR-100（百類彩色圖片分類）

**腳本**：`src/cifar100.py`

### 資料集

- 訓練集：50,000 張 32×32 彩色圖（RGB 3 通道）
- 測試集：10,000 張
- 類別：100 類（20 大類，每大類 5 小類），涵蓋動物、交通工具、日常物品等

### 模型架構

比 CIFAR-10 版多一組殘差層，以容納 100 個分類所需的特徵容量：

```
輸入 (32×32×3)
→ Conv2D(64, 3×3) + BN + ReLU
→ Layer1: 2× ResidualBlock(64→64)    →  32×32×64
→ Layer2: 2× ResidualBlock(64→128)   →  16×16×128
→ Layer3: 2× ResidualBlock(128→256)  →  8×8×256
→ Layer4: 2× ResidualBlock(256→512)  →  4×4×512
→ GlobalAveragePooling → Dropout(0.3) → Flatten
→ Dense(512→100)  →  100 個類別
```

### 超參數

| 項目 | 值 |
|------|----|
| Batch Size | 128 |
| Learning Rate | 0.001（CosineDecay 衰減） |
| Epochs | 30 |
| Optimizer | Adam |
| LR Scheduler | CosineDecay |
| Dropout | 0.3（FC 前） |

訓練集使用 RandomCrop、RandomHorizontalFlip 與 ColorJitter 資料增強。

### 執行

```bash
python src/cifar100.py
```

訓練紀錄輸出至 `logs/CIFAR100/`

---

## 如何讓模型跑在 GPU 上

本專案已在 `src/device.py` 內集中處理裝置選擇，TensorFlow 會自動偵測可用的 GPU 並分配運算。

### NVIDIA GPU（CUDA）

TensorFlow 自動偵測 NVIDIA GPU：

```python
gpus = tf.config.list_physical_devices('GPU')
# 若 gpus 非空，TF 即自動使用 GPU
```

只要安裝了 CUDA 驅動與對應的 TensorFlow 版本，所有運算會自動在 GPU 上執行，不需要手動搬移張量。

### Mac（Apple Silicon）GPU：Metal

在 Apple Silicon（M1/M2/M3…）的 macOS 上，TensorFlow 透過 `tensorflow-metal` 外掛使用 Metal GPU。

安裝方式：

```bash
pip install tensorflow-metal
```

安裝後，`tf.config.list_physical_devices('GPU')` 會回傳 Metal GPU，TF 會自動使用它。

你可以用以下方式確認：

```python
import tensorflow as tf

print(tf.config.list_physical_devices('GPU'))
```

若有輸出 Metal 裝置，即表示 GPU 加速已啟用。

### TF 資料管線：tf.data

本專案使用 `tf.data.Dataset` 取代 PyTorch 的 DataLoader，透過 `.prefetch(tf.data.AUTOTUNE)` 讓 GPU 計算與 CPU 資料預載平行進行：

```python
train_dataset = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(len(x_train))
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)
```

---

## 訓練紀錄

每次訓練結束後自動將結果匯出為 txt，依資料集分資料夾存放：

```
logs/
├── MNIST/
│   └── training_log_20260502_001118.txt
├── CIFAR10/
│   └── training_log_20260502_012345.txt
└── CIFAR100/
    └── training_log_20260502_023456.txt
```

報告表頭會包含 **電腦名稱（hostname）**、**完整 Python 版本**、**TensorFlow 版本**（`tf.__version__`）、**訓練裝置與裝置名稱**。若實際使用 **NVIDIA GPU（CUDA）** 訓練，會額外寫入 **CUDA 版本**。

透過頂部的 `ENABLE_LOGGING` 旗標控制是否匯出：

```python
ENABLE_LOGGING = True   # 改為 False 可關閉
```

---

## 環境需求

- Python 3.8+
- TensorFlow 2.12+
- tqdm

```bash
pip install -r requirements.txt
```

若使用 **Apple Silicon Mac**，額外安裝：

```bash
pip install tensorflow-metal
```

若需要 **NVIDIA CUDA** 支援，請至 [tensorflow.org](https://www.tensorflow.org/install/pip) 依照作業系統與 CUDA 版本選擇安裝指令。


## 作者資訊

- 姓名 Name: 王建葦 Albert W.
- 電子郵件 Email: albert@mail.jw-albert.tw

## 貢獻

歡迎提交 Issue 和 Pull Request 來改善這個專案

## 授權

本專案採用 MIT 授權條款