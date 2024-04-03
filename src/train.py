import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers
from models.cnn import CNN
from models.EfficientNet import Customed_EfficientNet
import numpy as np
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


class CustomModelCheckpoint(Callback):
    def __init__(self, filepath, save_freq):
        super(CustomModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.save_freq = save_freq

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_freq == 0:  # 에포크는 0부터 시작하므로 +1
            self.model.save(self.filepath.format(epoch=epoch + 1))

# 모델 정의


def main():
    # argparse를 사용하여 커맨드 라인 인자 처리
    parser = argparse.ArgumentParser(description='Train a CNN on the STL-10 dataset.')
    parser.add_argument('--model', type=str, default='CNN', help='Model to train.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train the model.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')

    args = parser.parse_args()
    (train_images, train_labels), (test_images, test_labels) = load_and_preprocess_data()
    # data augmentation
    datagen = create_data_augmentation_generator()
    train_generator = datagen.flow(train_images, train_labels, batch_size=args.batch_size)
    # scheduler
    lr_scheduler = LearningRateScheduler(scheduler)
    #args.model = 'CNN'
    if args.model == 'CNN':
        model = CNN()
    elif args.model == 'EfficientNet':
        model = Customed_EfficientNet()
    else:
        raise ValueError('Unknown model type: {}'.format(args.model))
    # 기존 모델이 존재할 경우, 불러와서 사용
    if os.path.exists('models/trained_models/' + args.model + '.h5'):
        model = tf.keras.models.load_model('models/trained_models/' + args.model + '.h5')

    # 로그 디렉토리 생성
    log_dir = "models/logs/fit/" + args.model + '/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    # 모델 컴파일
    model.compile(optimizer=optimizers.Adam(learning_rate=args.lr), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy']) # 출력층 activation='softmax'로 설정했기 때문에 from_logits=False
    # 커스텀 모델 체크포인트 콜백
    custom_checkpoint_callback = CustomModelCheckpoint('models/trained_models/' + args.model + '_epoch_{epoch}.h5', save_freq=5)

    # argparse를 사용하여 받은 epochs만큼 모델 학습
    model.fit(train_generator, epochs=args.epochs, validation_data=(test_images, test_labels), callbacks=[tensorboard_callback, lr_scheduler, custom_checkpoint_callback])

    # 모델 저장
    model.save('models/trained_models/' + args.model + '.h5')

if __name__ == '__main__':
    main()