import tensorflow as tf
import numpy as np
import time
from tqdm import tqdm
from logger import TrainingLogger
from device import (
    get_best_tf_device,
    get_device_display_info,
    get_mac_chip_info,
)

# 設為 False 可關閉訓練紀錄匯出
ENABLE_LOGGING = True


class CNN(tf.keras.Model):
    """兩層卷積 + 兩層全連接的 CNN，用於 MNIST 10 類分類。"""

    def __init__(self):
        super(CNN, self).__init__()
        # TF 預設 channels_last 格式：(batch, height, width, channels)
        # padding='same' 保持空間尺寸不變（等同 PyTorch 的 padding=2 for 5×5 kernel）
        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=5, padding='same')
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)
        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=5, padding='same')
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)
        # 攤平後 7×7×64 = 3136 個特徵
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(128)
        # 訓練時隨機關閉 50% 神經元，防止過擬合
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.fc2 = tf.keras.layers.Dense(10)

    def call(self, x, training=False):
        x = self.pool1(tf.nn.relu(self.conv1(x)))   # 32×28×28 → 32×14×14
        x = self.pool2(tf.nn.relu(self.conv2(x)))   # 64×14×14 → 64×7×7
        x = self.flatten(x)                          # 攤平成 (batch, 3136)
        x = tf.nn.relu(self.fc1(x))
        x = self.dropout(x, training=training)
        x = self.fc2(x)                              # 輸出 10 個 logits
        return x


@tf.function
def train_step(model, images, labels, loss_fn, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_fn(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, predictions


def train_epoch(model, dataset, loss_fn, optimizer, num_batches):
    """執行一個 epoch 的訓練，回傳平均 loss 與訓練準確率。"""
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataset, total=num_batches, desc="訓練"):
        loss, predictions = train_step(model, images, labels, loss_fn, optimizer)

        total_loss += float(loss)
        predicted = tf.argmax(predictions, axis=1, output_type=tf.int32)
        correct += int(tf.reduce_sum(tf.cast(predicted == tf.cast(labels, tf.int32), tf.int32)).numpy())
        total += int(labels.shape[0])

    avg_loss = total_loss / num_batches
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def test(model, dataset, num_batches):
    """在測試集上評估模型，回傳準確率。"""
    correct = 0
    total = 0

    # TF 不需要 no_grad 情境：只要不在 GradientTape 內，就不會記錄梯度
    for images, labels in tqdm(dataset, total=num_batches, desc="測試"):
        predictions = model(images, training=False)
        predicted = tf.argmax(predictions, axis=1, output_type=tf.int32)
        correct += int(tf.reduce_sum(tf.cast(predicted == tf.cast(labels, tf.int32), tf.int32)).numpy())
        total += int(labels.shape[0])

    accuracy = 100.0 * correct / total
    return accuracy


if __name__ == '__main__':
    device = get_best_tf_device()
    device_type, device_detail = get_device_display_info(device)
    print(f"使用設備: {device} ({device_type})")
    if device_detail:
        print(f"裝置資訊: {device_detail}")
    chip = get_mac_chip_info()
    if chip.is_macos:
        print(f"Mac 晶片辨識: machine={chip.machine}, apple_silicon={chip.is_apple_silicon}")

    logger = TrainingLogger(enabled=ENABLE_LOGGING, device=device)

    batch_size = 128
    learning_rate = 0.001
    epochs = 50

    # 標準化參數來自 MNIST 全資料集的均值與標準差
    mean = np.float32(0.1307)
    std = np.float32(0.3081)

    print("\n加載 MNIST 數據...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # 加入 channel 維度：(N, 28, 28) → (N, 28, 28, 1)，轉 float32 並正規化
    x_train = (x_train[..., np.newaxis].astype(np.float32) / 255.0 - mean) / std
    x_test = (x_test[..., np.newaxis].astype(np.float32) / 255.0 - mean) / std

    num_train_batches = int(np.ceil(len(x_train) / batch_size))
    num_test_batches = int(np.ceil(len(x_test) / batch_size))

    # prefetch 讓 GPU 計算與 CPU 資料預載平行進行，減少等待
    train_dataset = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(len(x_train))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_dataset = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    model = CNN()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    print("\n開始訓練...\n")
    start_time = time.time()
    logger.start()

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_dataset, loss_fn, optimizer, num_train_batches)
        test_acc = test(model, test_dataset, num_test_batches)
        elapsed = time.time() - start_time

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
        logger.log_epoch(epoch + 1, epochs, train_loss, train_acc, test_acc, elapsed)

    total_time = time.time() - start_time
    print(f"\n訓練完成！耗時: {total_time:.2f} 秒")

    logger.finish(total_time)
    logger.export(title="MNIST CNN 訓練紀錄", output_dir="./logs/MNIST")

    model.save_weights('mnist_cnn.weights.h5')
    print("模型已保存為 mnist_cnn.weights.h5")
