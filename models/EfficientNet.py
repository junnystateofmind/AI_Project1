import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models


def Customed_EfficientNet(input_shape=(96, 96, 3), num_classes=10):
    inputs = layers.Input(shape=input_shape)

    # base_model을 추론 모드로 설정합니다.
    x = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(96, 96, 3))(inputs)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='leaky_relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model


# 모델 인스턴스를 생성합니다.
custom_model = Customed_EfficientNet(input_shape=(96, 96, 3), num_classes=10)
# 모델 요약을 출력합니다.
custom_model.summary()
