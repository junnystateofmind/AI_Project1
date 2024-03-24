import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers
from ..models.cnn import CNN

# 데이터 로드 및 전처리
def load_and_preprocess_data():
    # STL-10 사용
    (train_images, train_labels), (test_images, test_labels) = datasets.stl10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    return train_images, train_labels, test_images, test_labels


# 모델 정의


def main():
    (train_images, train_labels), (test_images, test_labels) = load_and_preprocess_data()
    model = CNN()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

    # 모델 저장
    model.save('model/my_model')

if __name__ == '__main__':
    main()
