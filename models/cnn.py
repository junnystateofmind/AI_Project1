import tensorflow as tf
from tensorflow.keras import layers, models

# Overview
# 10 classes: airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck.
# Images are 96x96 pixels, color.
# 500 training images (10 pre-defined folds), 800 test images per class.
# 100000 unlabeled images for unsupervised learning. These examples are extracted from a similar but broader distribution of images. For instance, it contains other types of animals (bears, rabbits, etc.) and vehicles (trains, buses, etc.) in addition to the ones in the labeled set.
# Images were acquired from labeled examples on ImageNet.
def residual_block(x, filters):
    """Depthwise 및 Pointwise Convolution을 사용한 잔차 블록."""
    # Shortcut pathway, 1x1 convolution
    shortcut = layers.Conv2D(filters, (1, 1), padding='same')(x)
    shortcut = layers.BatchNormalization()(shortcut)

    # Depthwise Convolution
    x = layers.DepthwiseConv2D((3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    # Pointwise Convolution
    x = layers.Conv2D(filters, (1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    # Add shortcut to the main path before activation
    x = layers.add([x, shortcut])
    x = layers.Activation('linear')(x)

    return x

def CNN(input_shape=(96, 96, 3), num_classes=10):
    inputs = layers.Input(shape=input_shape)

    # First Convolutional Layer
    x = layers.Conv2D(64, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Residual Blocks with increased filter numbers
    x = residual_block(x, 128)
    x = layers.MaxPooling2D((2, 2))(x)

    x = residual_block(x, 256)
    x = layers.MaxPooling2D((2, 2))(x)

    x = residual_block(x, 2048)
    x = layers.MaxPooling2D((2, 2))(x)

    x = residual_block(x, 4096)
    x = layers.MaxPooling2D((2, 2))(x)

    x = residual_block(x, 4096)
    x = layers.MaxPooling2D((2, 2))(x)

    x = residual_block(x, 256)

    # Global Average Pooling followed by Classification Layer
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.7)(x)  # Dropout added to combat overfitting
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model



def SimpleCNN(input_shape=(96, 96, 3), num_classes=10):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape), # 94x94x32
        layers.MaxPooling2D((2, 2)), # 47x47x32
        layers.Conv2D(64, (3, 3), activation='relu'), # 45x45x64
        layers.MaxPooling2D((2, 2)), # 22x22x64
        layers.Conv2D(64, (3, 3), activation='relu'), # 20x20x64
        layers.Conv2D(64, (3, 3), activation='relu'), # 18x18x64
        layers.Flatten(), # 20736
        layers.Dense(64, activation='relu'), #
        layers.Dense(num_classes)
    ])
    return model
# 모델 요약

model = CNN()
model.summary()
