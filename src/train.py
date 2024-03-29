import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers
# models/cnn 모듈에서 CNN 함수를 불러옴
from models.cnn import CNN
import numpy as np
import os

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
    data_path = os.path.join(current_path, '..', 'data')
    train_images = load_stl10_dataset(data_path + '/train_X.bin')
    train_labels = np.fromfile(data_path + '/train_y.bin', dtype=np.uint8) - 1
    test_images = load_stl10_dataset(data_path + '/test_X.bin')
    test_labels = np.fromfile(data_path + '/test_y.bin', dtype=np.uint8) - 1

    # 데이터 전처리
    train_images = preprocess_images(train_images)
    test_images = preprocess_images(test_images)

    return (train_images, train_labels), (test_images, test_labels)
# 모델 정의


def main():
    (train_images, train_labels), (test_images, test_labels) = load_and_preprocess_data()
    model = CNN()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

    # 모델 저장
    model.save('AI_Project1/models/trained_models/cnn_model.h5')

if __name__ == '__main__':
    main()
