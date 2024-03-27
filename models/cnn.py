import tensorflow as tf
from tensorflow.keras import layers, models

# Overview
# 10 classes: airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck.
# Images are 96x96 pixels, color.
# 500 training images (10 pre-defined folds), 800 test images per class.
# 100000 unlabeled images for unsupervised learning. These examples are extracted from a similar but broader distribution of images. For instance, it contains other types of animals (bears, rabbits, etc.) and vehicles (trains, buses, etc.) in addition to the ones in the labeled set.
# Images were acquired from labeled examples on ImageNet.

def CNN(input_shape=(96, 96, 3), num_classes=10):
    model = models.Sequential([
        layers.Conv2D(kernel_size=(3,3), filters=64, activation='relu', input_shape=input_shape), # 로컬 패턴과 질감, 기본적인 형태를 감지하는 데 중점을 두기 위해 64개의 커널을 사용, 94x94x64
        layers.MaxPooling2D((2, 2)), # 풀링 레이어를 추가, 차원은 47x47x64
        layers.Conv2D(kernel_size=(3,3), filters=256, activation='relu'), # 커널 수 256개로 늘려 세부적인 패턴을 감지, 45x45x256
        # depthwise separable convolution을 통해 모델의 크기를 줄이고, 계산량을 줄이기 위해 사용
        layers.DepthwiseConv2D(kernel_size=(3,3), activation='relu'), # 43x43x256
        # pointwise convolution을 통해 depthwise separable convolution의 출력을 다시 합쳐줌
        layers.Conv2D(filters=256, kernel_size=(1,1), activation='relu'), # 43x43x256
        # 풀링 레이어를 추가
        layers.MaxPooling2D((2, 2)), # 21x21x256
        # 한번 더 depthwise separable convolution을 적용
        layers.DepthwiseConv2D(kernel_size=(3,3), activation='relu'), # 19x19x256
        layers.Conv2D(filters=256, kernel_size=(1,1), activation='relu'), # 19x19x256
        layers.GlobalAveragePooling2D(), # 256
        layers.Dense(128, activation='relu'), # 128
        layers.Dense(num_classes) # 10
    ])
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

