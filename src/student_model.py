import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import KLDivergence, CategoricalCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import argparse
from models.EfficientNet import Customed_EfficientNetB0, Customed_EfficientNetB3, Customed_EfficientNetB4, Customed_EfficientNetB6, Customed_EfficientNetB7
from models.cnn import CNN
from src.train import load_and_preprocess_data, create_data_augmentation_generator, LearningRateScheduler, scheduler, CustomModelCheckpoint

def load_teacher_model(path):
    teacher_model = load_model(path, compile=False)
    teacher_model.trainable = False
    teacher_model.summary()
    # 모델의 이름 변경
    teacher_model._name = 'teacher_model'
    # model.get_layer(name='predictions').name='teacher_predictions'
    return teacher_model

def create_student_model(input_shape, num_classes, student_base_model):
    student_model = student_base_model(input_shape=input_shape, num_classes=num_classes)
    student_model.summary()
    # 모델의 이름 변경
    student_model._name = 'student_model'
    # model.get_layer(name='predictions').name='student_predictions'
    return student_model


def create_distillation_model(teacher_model, student_model, input_shape):
    inputs = keras.Input(shape=input_shape, name='distillation_input')

    teacher_outputs = teacher_model(inputs)
    student_outputs = student_model(inputs)

    # 명시적으로 레이어 이름 추가
    teacher_outputs = layers.Activation('softmax', name='distillation_teacher_output')(teacher_outputs)
    student_outputs = layers.Activation('softmax', name='distillation_student_output')(student_outputs)

    # 모델 이름 명시적 지정
    model = models.Model(inputs=inputs, outputs=[teacher_outputs, student_outputs], name='unique_distillation_model')
    return model


def main():
    parser = argparse.ArgumentParser(description='Knowledge Distillation Training Script')
    parser.add_argument('--teacher_model_path', type=str, default='models/trained_models/finetuning_EfficientNetB4.h5', help='Path to the teacher model.')
    parser.add_argument('--input_shape', type=tuple, default=(96, 96, 3), help='Input shape of the images.')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes in the dataset.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('--student_base_model', type=str, default='EfficientNetB0', help='Base model for the student model.')
    args = parser.parse_args()

    teacher_model = load_teacher_model(args.teacher_model_path)
    student_model = create_student_model(args.input_shape, args.num_classes, globals()[args.student_base_model])
    distillation_model = create_distillation_model(teacher_model, student_model, args.input_shape)
    print("distillation_model summary")
    distillation_model.summary()

    distillation_model.compile(
        optimizer=keras.optimizers.Adam(),
        loss={
            'distillation_teacher_output': keras.losses.KLDivergence(),
            'distillation_student_output': keras.losses.SparseCategoricalCrossentropy()
        },
        loss_weights={
            'distillation_teacher_output': 0.6,
            'distillation_student_output': 0.4
        },
        metrics={
            'distillation_student_output': ['accuracy']
        }
    )

    # Load and preprocess data
    (train_images, train_labels), (test_images, test_labels) = load_and_preprocess_data()

    # callbacks
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir='logs/distillation', histogram_freq=1)
    custom_checkpoint_callback = CustomModelCheckpoint('models/trained_models/distillation_model/distillation_model_epoch_{epoch}.h5', save_freq=10)

    distillation_model.fit(
        train_images, {'distillation_teacher_output': train_labels, 'distillation_student_output': train_labels},
        validation_data=(test_images, {'distillation_teacher_output': test_labels, 'distillation_student_output': test_labels}),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[tensorboard_callback, custom_checkpoint_callback, LearningRateScheduler(scheduler)]
    )


if __name__ == '__main__':
    main()
