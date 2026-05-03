# GPU-Learning-Tensorflow — Agent 指引

本文件給 **Cursor Agent 、 Claude 與其他協作者** 快速理解專案範圍、程式結構與修改時的注意事項。

---

## 任務說明

透過常見公開資料集（MNIST、CIFAR-10、CIFAR-100）在不同運算裝置上訓練 CNN／ResNet 模型，比較 **CPU、NVIDIA CUDA、Apple Silicon Metal** 等環境下的訓練行為；訓練過程可選擇將紀錄匯出為文字檔。

---

## 技術棧

- **Python 3.10+**
- **TensorFlow 2.12+**（版本見 `requirements.txt`）
- **tqdm**（進度列）
- 資料集第一次執行會下載至 `~/.keras/datasets/`（見 `.gitignore`，通常不提交資料）

---

## 目錄結構（精簡）

| 路徑 | 說明 |
|------|------|
| `src/mnist.py` | MNIST：簡單 CNN 訓練腳本 |
| `src/cifar10.py` | CIFAR-10：ResNet（3 層）訓練腳本 |
| `src/cifar100.py` | CIFAR-100：ResNet（4 層）訓練腳本 |
| `src/device.py` | 統一選擇 `/GPU:0` / `/CPU:0`，以及裝置顯示資訊 |
| `src/logger.py` | `TrainingLogger`：訓練紀錄與 txt 匯出（含主機名、完整 Python 版本、`tf.__version__`、裝置資訊；CUDA 訓練時含 CUDA 版本） |
| `logs/MNIST/`、`logs/CIFAR10/`、`logs/CIFAR100/` | 訓練輸出紀錄（執行時產生，勿隨意納入版本庫） |
| `docs/MNIST.md` | MNIST 模型詳細說明 |
| `docs/CIFAR10.md` | CIFAR-10 模型詳細說明 |
| `docs/CIFAR100.md` | CIFAR-100 模型詳細說明 |
| `README.md` | 使用者總覽說明（含 Metal/CUDA 章節） |
| `CNN_GUIDE.md` | CNN 概念指南 |
| `mnist.sh` / `cifar10.sh` / `cifar100.sh` | 啟用 `venv` 後執行對應腳本（專案根目錄需有 `venv/`） |

---

## 裝置選擇邏輯（勿重複發明輪子）

邏輯集中在 **`src/device.py`**：

1. `tf.config.list_physical_devices('GPU')` 非空 → **`/GPU:0`**（NVIDIA CUDA 或 Apple Metal）
2. 否則 → **`/CPU:0`**

TensorFlow 自動將所有運算分配至可用 GPU，`get_best_tf_device()` 的回傳值主要用於顯示與 Logger 紀錄。

`TrainingLogger` 應傳入與訓練相同的 `device` 字串，以便匯出檔與實際運算一致。

---

## 執行方式

**直接執行（建議在虛擬環境內）：**

```bash
python src/mnist.py
python src/cifar10.py
python src/cifar100.py
```

**或使用腳本（需先有 `venv/`）：**

```bash
bash mnist.sh
bash cifar10.sh
bash cifar100.sh
```

安裝依賴：

```bash
pip install -r requirements.txt
# Apple Silicon Mac 額外安裝：
pip install tensorflow-metal
```

---

## 修改程式時的約束

1. **裝置相關**：新增或調整「選 GPU／CPU」時，請優先擴充 **`src/device.py`**，再在 `mnist.py` / `cifar10.py` / `cifar100.py` 呼叫，避免各檔各寫一套。
2. **訓練紀錄**：輸出格式、主機名、**Python 完整版本**、**TensorFlow 版本**、裝置字串與 **CUDA 版本（僅 `cuda` 裝置）** 在 **`src/logger.py`**；若只改報告欄位，通常只動此檔即可。
3. **語言與風格**：使用者可讀字串與註解維持 **繁體中文**（與現有程式一致）；不必要的重構、無關檔案不要動。
4. **版本庫**：不要主動 commit 下載的權重（`.weights.h5`）、以及 **`logs/`** 內訓練產生的 txt（除非使用者明確要求）。
5. **新增模型**：新資料集的訓練腳本應遵循現有架構：`get_best_tf_device` → `TrainingLogger` → 資料前處理 → `tf.data.Dataset` → model → optimizer → lr_schedule → 訓練迴圈 → `logger.export`。

---

## 相關文件

- 使用說明與 Metal/CUDA 注意事項：**`README.md`**
- 依賴版本：**`requirements.txt`**
- MNIST 模型詳細說明：**`docs/MNIST.md`**
- CIFAR-10 模型詳細說明：**`docs/CIFAR10.md`**
- CIFAR-100 模型詳細說明：**`docs/CIFAR100.md`**
