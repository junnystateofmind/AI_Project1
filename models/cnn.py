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
    # 입력과 동일한 차원의 출력을 생성하기 위해, 필터 수에 맞는 1x1 합성곱 레이어로 shortcut 경로 생성
    shortcut = layers.Conv2D(filters, (1, 1), padding='same')(x)

    # Depthwise Convolution
    x = layers.DepthwiseConv2D((3, 3), activation='leaky_relu', padding='same')(x)

    # Pointwise Convolution
    x = layers.Conv2D(filters, (1, 1), activation='leaky_relu', padding='same')(x)

    # 입력(x)과 shortcut을 더함
    x = layers.add([x, shortcut])
    x = layers.Activation('leaky_relu')(x)

    return x


def CNN(input_shape=(96, 96, 3), num_classes=10):
    inputs = layers.Input(shape=input_shape)

    # 첫 번째 합성곱 레이어
    x = layers.Conv2D(64, (3, 3), activation='leaky_relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    # 잔차 블록 추가
    x = residual_block(x, 256)
    x = layers.MaxPooling2D((2, 2))(x)

    x = residual_block(x, 512)
    x = layers.MaxPooling2D((2, 2))(x)

    x = residual_block(x, 1024)
    x = layers.MaxPooling2D((2, 2))(x)

    x = residual_block(x, 2048)
    x = layers.MaxPooling2D((2, 2))(x)

    # # 전역 평균 풀링과 분류 레이어
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='leaky_relu')(x)
    outputs = layers.Dense(num_classes)(x)

    # 각주 부분은 GlobalAveragePooling2D를 사용하지 않고 Flatten과 Dense 레이어를 사용한 경우
    # x = layers.Flatten()(x)
    # x = layers.Dense(64, activation='relu')(x)
    # outputs = layers.Dense(num_classes)(x)

    model = models.Model(inputs=inputs, outputs=outputs)
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

