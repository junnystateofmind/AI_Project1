import math

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers
from models.cnn import CNN
import numpy as np
import time
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler, EarlyStopping, Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import datetime
import os
import sys
import argparse

# 데이터 로드 및 전처리
# 바이너리 파일에서 데이터 로딩 함수
def load_stl10_dataset(data_path):
    with open(data_path, 'rb') as file:
        data = np.fromfile(file, dtype=np.uint8)
    images = np.reshape(data, (-1, 3, 96, 96))
    images = np.transpose(images, (0, 3, 2, 1))  # TensorFlow에서 사용하기 위해 축 변경
    return images

# 데이터 전처리 함수
def preprocess_images(images):
    images = tf.cast(images, tf.float32)
    images = images / 255.0  # 정규화
    return images

def load_and_preprocess_data():
    # 데이터 로드, 현재 path
    current_path = os.path.dirname(os.path.abspath(__file__)) # 현재 파일의 절대 경로
    # data path는 현재 파일의 상위 디렉토리의 data 폴더
    data_path = os.path.join(current_path, '..', 'data/stl10_binary')
    train_images = load_stl10_dataset(data_path + '/train_X.bin')
    train_labels = np.fromfile(data_path + '/train_y.bin', dtype=np.uint8) - 1
    test_images = load_stl10_dataset(data_path + '/test_X.bin')
    test_labels = np.fromfile(data_path + '/test_y.bin', dtype=np.uint8) - 1

    # 데이터 전처리
    train_images = preprocess_images(train_images)
    test_images = preprocess_images(test_images)

    return (train_images, train_labels), (test_images, test_labels)
# data augmentation
def create_data_augmentation_generator():
    return ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
# scheduler
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

def custom_scheduler(epoch, lr):
    wave_height = 0.2
    if epoch < 30:
        return lr
    else:
        decay_rate = 0.05
        return lr * tf.exp(-decay_rate) * (1 + tf.cos(epoch * tf.constant(np.pi) / 5) * wave_height)

def cosine_annealing_with_warmup(epoch, initial_learning_rate, first_decay_steps = 10, t_mul=2.0, m_mul=1.0, alpha_zero=0.0):
    t_curr = epoch % first_decay_steps
    if epoch < first_decay_steps:
        alpha = alpha_zero + (initial_learning_rate - alpha_zero) * (1 + np.cos(np.pi * t_curr / first_decay_steps)) / 2
    else:
        if t_curr == 0:
            first_decay_steps *= t_mul
            initial_learning_rate *= m_mul
        alpha = initial_learning_rate + (initial_learning_rate - alpha_zero) * (1 + np.cos(np.pi * t_curr / first_decay_steps)) / 2
    return alpha


# 아래 콜백은 특정 에포크에 도달했을 때 상위 레이어의 학습 가능 상태를 변경
class UnfreezeLayersCallback(Callback):
    def __init__(self, unfreeze_at_epoch, model):
        super(UnfreezeLayersCallback, self).__init__()
        self.unfreeze_at_epoch = unfreeze_at_epoch
        self.model = model

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == self.unfreeze_at_epoch:
            # 특정 에포크에 도달했을 때 상위 레이어의 학습 가능 상태를 변경
            for layer in self.model.layers[-10:]:
                layer.trainable = True
            # 레이어 상태 변경 후 모델을 다시 컴파일
            self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            print(f"Unfreezing top layers at epoch {epoch+1}")


class CustomModelCheckpoint(Callback):
    def __init__(self, filepath, save_freq):
        super(CustomModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.save_freq = save_freq

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_freq == 0:
            # 모든 텐서를 NumPy 배열로 변환
            weights = [w.numpy() if isinstance(w, tf.Tensor) else w for w in self.model.get_weights()]
            self.model.set_weights(weights)
            # 모델 저장
            self.model.save(self.filepath.format(epoch=epoch + 1))




# 모델 정의


def main():
    # argparse를 사용하여 커맨드 라인 인자 처리
    parser = argparse.ArgumentParser(description='Train a CNN on the STL-10 dataset.')
    parser.add_argument('--model', type=str, default='CNN', help='Model to train.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train the model.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--start_epoch', type=int, default=0, help='Epoch to start training from.')
    parser.add_argument('--unfreeze_at_epoch', type=int, default=0, help='Epoch to unfreeze top layers at.')
    parser.add_argument('--scheduler', type = str, default = 'LearningRateScheduler', help = 'Learning rate scheduler to use.')

    args = parser.parse_args()
    (train_images, train_labels), (test_images, test_labels) = load_and_preprocess_data()
    # data augmentation
    datagen = create_data_augmentation_generator()
    train_generator = datagen.flow(train_images, train_labels, batch_size=args.batch_size)
    # scheduler
    if args.scheduler == 'LearningRateScheduler':
        lr_scheduler = LearningRateScheduler(scheduler)
    elif args.scheduler == 'custom':
        lr_scheduler = LearningRateScheduler(custom_scheduler)
    elif args.scheduler == 'CosineAnnealingWarmUpRestarts':
        lr_scheduler = LearningRateScheduler(lambda epoch: cosine_annealing_with_warmup(epoch, args.lr, first_decay_steps=10))
    else:
        raise ValueError('Unknown scheduler type: {}'.format(args.scheduler))
    #args.model = 'CNN'
    if args.model == 'CNN':
        model = CNN()
    elif args.model == 'EfficientNetB0':
        from models.EfficientNet import Customed_EfficientNetB0
        model = Customed_EfficientNetB0()
    elif args.model == 'EfficientNetB3':
        from models.EfficientNet import Customed_EfficientNetB3
        model = Customed_EfficientNetB3()
    elif args.model == 'EfficientNetB4':
        from models.EfficientNet import Customed_EfficientNetB4
        model = Customed_EfficientNetB4()
    elif args.model == 'EfficientNetB6':
        from models.EfficientNet import Customed_EfficientNetB6
        model = Customed_EfficientNetB6()
    elif args.model == 'EfficientNetB7':
        from models.EfficientNet import Customed_EfficientNetB7
        model = Customed_EfficientNetB7()
    else:
        raise ValueError('Unknown model type: {}'.format(args.model))
    # 기존 모델이 존재할 경우, 불러와서 사용
    if args.start_epoch > 0 and os.path.exists('models/trained_models/' + args.model + '/' + args.model + '_epoch_' + str(args.start_epoch) + '.h5'):
        model = tf.keras.models.load_model('models/trained_models/' + args.model + '/' + args.model + '_epoch_' + str(args.start_epoch) + '.h5')

    # 로그 디렉토리 생성
    log_dir = "models/logs/fit/" + args.model + '/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    # 모델 컴파일
    model.compile(optimizer=optimizers.Adam(learning_rate=args.lr), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy']) # 출력층 activation='softmax'로 설정했기 때문에 from_logits=False
    # 커스텀 모델 체크포인트 콜백
    # 메인 함수 내에서 콜백 사용
    custom_checkpoint_callback = CustomModelCheckpoint('models/trained_models/' + args.model + '/' + args.model + '_' +  time.strftime("_%Y%m%d-%H%M%S") + '_epoch_' + str(args.epochs) + '.h5', save_freq=10)

    # argparse를 사용하여 받은 epochs만큼 모델 학습
    # fine_tuning_models = ['EfficientNetB0', 'EfficientNetB3', 'EfficientNetB4']
    # if args.model in fine_tuning_models:
    #     unfreeze_callback = UnfreezeLayersCallback(unfreeze_at_epoch=args.unfreeze_at_epoch, model=model)
    #     callbacks = [tensorboard_callback, lr_scheduler, custom_checkpoint_callback, unfreeze_callback]
    # else:
    #     callbacks = [tensorboard_callback, lr_scheduler, custom_checkpoint_callback]

    callbacks = [tensorboard_callback, lr_scheduler, custom_checkpoint_callback]
    model.fit(train_generator, epochs=args.epochs, validation_data=(test_images, test_labels), callbacks=callbacks)

    # 모델 저장
    model.save('models/trained_models/' + args.model + '/' + args.model + '_' +  time.strftime("_%Y%m%d-%H%M%S") + '_epoch_' + str(args.epochs) + '.h5')

if __name__ == '__main__':
    main()