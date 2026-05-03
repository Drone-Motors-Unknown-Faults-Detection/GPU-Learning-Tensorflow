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

# L2 正則化強度（等同 PyTorch Adam 的 weight_decay=1e-4）
_L2 = tf.keras.regularizers.l2(1e-4)


class ResidualBlock(tf.keras.layers.Layer):
    """單一殘差塊：兩層 3×3 卷積 + shortcut 連接，避免深層網路梯度消失。"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            out_channels, 3, strides=stride, padding='same', use_bias=False, kernel_regularizer=_L2
        )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(
            out_channels, 3, strides=1, padding='same', use_bias=False, kernel_regularizer=_L2
        )
        self.bn2 = tf.keras.layers.BatchNormalization()

        # 當尺寸或通道數改變時，shortcut 需要 1×1 卷積對齊維度
        self.need_projection = (stride != 1 or in_channels != out_channels)
        if self.need_projection:
            self.shortcut_conv = tf.keras.layers.Conv2D(
                out_channels, 1, strides=stride, use_bias=False, kernel_regularizer=_L2
            )
            self.shortcut_bn = tf.keras.layers.BatchNormalization()

    def call(self, x, training=False):
        out = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        out = self.bn2(self.conv2(out), training=training)

        if self.need_projection:
            shortcut = self.shortcut_bn(self.shortcut_conv(x), training=training)
        else:
            shortcut = x

        return tf.nn.relu(out + shortcut)  # 殘差相加


class ResNet(tf.keras.Model):
    """簡化版 ResNet，三組殘差層處理 CIFAR-10 的 32×32 彩色圖片（3 通道）。"""

    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        # 初始卷積：3 通道輸入 → 64 特徵圖，不縮小尺寸
        self.conv1 = tf.keras.layers.Conv2D(64, 3, strides=1, padding='same', use_bias=False, kernel_regularizer=_L2)
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.layer1 = self._make_layer(64,  64,  blocks=2, stride=1)   # 32×32
        self.layer2 = self._make_layer(64,  128, blocks=2, stride=2)   # 16×16
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)   # 8×8

        # GlobalAveragePooling2D 讓輸出固定為 1×1，不受輸入尺寸影響
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.fc = tf.keras.layers.Dense(num_classes, kernel_regularizer=_L2)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return layers

    def call(self, x, training=False):
        x = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        for block in self.layer1:
            x = block(x, training=training)
        for block in self.layer2:
            x = block(x, training=training)
        for block in self.layer3:
            x = block(x, training=training)
        x = self.avgpool(x)                        # (batch, 256)
        x = self.dropout(x, training=training)
        x = self.fc(x)                             # (batch, num_classes) logits
        return x


@tf.function
def train_step(model, images, labels, loss_fn, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_fn(labels, predictions)
        # 加入 kernel_regularizer 的 L2 損失（等同 weight_decay）
        if model.losses:
            loss += tf.add_n(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, predictions


def train_epoch(model, dataset, loss_fn, optimizer, num_batches):
    """執行一個 epoch 的訓練，回傳平均 loss 與訓練準確率。"""
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataset, total=num_batches, desc="訓練", leave=False):
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

    for images, labels in tqdm(dataset, total=num_batches, desc="測試", leave=False):
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

    batch_size = 256
    learning_rate = 0.001
    epochs = 20

    # CIFAR-10 官方統計值（各通道 mean/std）
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
    std  = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)

    print("\n加載 CIFAR-10 數據...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    # Keras 回傳的 labels 為 (N, 1)，攤平成 (N,)
    y_train = y_train.reshape(-1)
    y_test  = y_test.reshape(-1)

    # 測試集：只做正規化（不做增強）
    x_train_float = x_train.astype(np.float32) / 255.0
    x_test_norm   = (x_test.astype(np.float32) / 255.0 - mean) / std

    num_train_batches = int(np.ceil(len(x_train) / batch_size))
    num_test_batches  = int(np.ceil(len(x_test)  / batch_size))

    # 訓練集資料增強：隨機裁切 + 水平翻轉 + 色彩抖動 + 正規化
    # 順序與 torchvision 一致：先在 [0,1] float 影像上增強，再正規化
    def augment_train(image, label):
        # RandomCrop(32, padding=4)：pad 至 40×40 後隨機裁回 32×32
        image = tf.image.resize_with_crop_or_pad(image, 40, 40)
        image = tf.image.random_crop(image, [32, 32, 3])
        image = tf.image.random_flip_left_right(image)
        # ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
        image = tf.clip_by_value(image, 0.0, 1.0)
        # Normalize
        image = (image - mean) / std
        return image, label

    train_dataset = (
        tf.data.Dataset.from_tensor_slices((x_train_float, y_train))
        .shuffle(len(x_train))
        .map(augment_train, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_dataset = (
        tf.data.Dataset.from_tensor_slices((x_test_norm, y_test))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    model = ResNet(num_classes=10)
    # label_smoothing 讓模型不過度自信，緩解 train loss 壓到極低但 test acc 停滯的問題
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, label_smoothing=0.1)
    # CosineDecay 平滑衰減，避免 StepLR 在某 epoch 造成 test acc 驟降的問題
    total_steps = epochs * num_train_batches
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate, decay_steps=total_steps
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

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
    logger.export(title="CIFAR-10 ResNet 訓練紀錄", output_dir="./logs/CIFAR10")

    model.save_weights('cifar10_resnet.weights.h5')
    print("模型已保存為 cifar10_resnet.weights.h5")
