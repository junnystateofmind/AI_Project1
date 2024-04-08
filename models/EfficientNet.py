import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB3, EfficientNetB4, EfficientNetB6, EfficientNetB7
from tensorflow.keras import layers, models


def Customed_EfficientNetB0(input_shape=(96, 96, 3), num_classes=10):
    inputs = layers.Input(shape=input_shape)

    # base_model 객체를 생성합니다.
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape)

    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64)(x)
    x = layers.LeakyReLU()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def Customed_EfficientNetB3(input_shape=(96, 96, 3), num_classes=10):
    inputs = layers.Input(shape=input_shape)

    # base_model 객체를 생성합니다.
    base_model = EfficientNetB3(include_top=False, weights='imagenet', input_shape=input_shape)

    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64)(x)
    x = layers.LeakyReLU()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model


def Customed_EfficientNetB4(input_shape=(96, 96, 3), num_classes=10):
    inputs = layers.Input(shape=input_shape)

    # base_model 객체를 생성합니다.
    base_model = EfficientNetB4(include_top=False, weights='imagenet', input_shape=input_shape)

    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64)(x)
    x = layers.LeakyReLU()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def Customed_EfficientNetB6(input_shape=(96, 96, 3), num_classes=10):
    inputs = layers.Input(shape=input_shape)

    # base_model 객체를 생성합니다.
    base_model = EfficientNetB6(include_top=False, weights='imagenet', input_shape=input_shape)

    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64)(x)
    x = layers.LeakyReLU()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def Customed_EfficientNetB7(input_shape=(96, 96, 3), num_classes=10):
    inputs = layers.Input(shape=input_shape)

    # base_model 객체를 생성합니다.
    base_model = EfficientNetB7(include_top=False, weights='imagenet', input_shape=input_shape)

    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64)(x)
    x = layers.LeakyReLU()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model