#해당 파일은 학습한 모델의 가중치를 바탕으로 학습을 위해 모델을 unfreeze하는 코드입니다.
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers
from models.cnn import CNN
from models.EfficientNet import Customed_EfficientNetB0, Customed_EfficientNetB3, Customed_EfficientNetB4, Customed_EfficientNetB6, Customed_EfficientNetB7
import numpy as np
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler, EarlyStopping, Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import datetime
import os
import sys
import argparse

# 학습한 모델 불러오기
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# 모델 unfreeze

def unfreeze_model(model, num_unfreeze_layers):
    # 모델의 레이어를 얼마나 unfreeze할지 결정
    for layer in model.layers[:-num_unfreeze_layers]:
        layer.trainable = False
    return model

# 모델 컴파일


# unfreeze scheduler
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

# 데이터 로드 및 전처리
from .train import load_and_preprocess_data, create_data_augmentation_generator, CustomModelCheckpoint
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='EfficientNetB4', help='Model to train')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to evaluate the model')
    parser.add_argument('--start_epoch', type=int, default=0, help='Epoch to start training from')
    parser.add_argument('--num_unfreeze_layers', type=int, default=0, help='Number of layers to unfreeze')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--log_dir', type=str, default='logs/unfreeze_train', help='Directory for TensorBoard logs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')

    args = parser.parse_args()

    (train_images, train_labels), (test_images, test_labels) = load_and_preprocess_data()
    # data augmentation
    datagen = create_data_augmentation_generator()
    train_generator = datagen.flow(train_images, train_labels, batch_size=args.batch_size)
    # scheduler
    lr_scheduler = LearningRateScheduler(scheduler)

    # 모델 로드
    model = load_model('models/trained_models/'+args.model+'/'+args.model+'_epoch_'+str(args.start_epoch)+'.h5')

    # 모델 unfreeze
    model = unfreeze_model(model, args.num_unfreeze_layers)

    # 모델 컴파일
    model.compile(optimizer=optimizers.Adam(learning_rate=args.learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    # 커스텀 모델 체크포인트 콜백
    custom_checkpoint_callback = CustomModelCheckpoint('models/trained_models/'+args.model+'/'+args.model+'_unfreeze_epoch_{epoch}.h5', save_freq=10)
    # TensorBoard 로그 디렉토리 설정
    log_dir = os.path.join(args.log_dir, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    # 모델 훈련
    callback = [tensorboard_callback, lr_scheduler, custom_checkpoint_callback]
    model.fit(train_generator, epochs=args.epochs, validation_data=(test_images, test_labels), callbacks=callback)

    # 저장
    model.save('models/trained_models/'+args.model+'/'+args.model+'_unfreeze_epoch_'+str(args.epochs)+'.h5')


if __name__ == '__main__':
    main()

