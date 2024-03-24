import tensorflow as tf
from tensorflow.keras import datasets

# 데이터 로드 및 전처리
def load_and_preprocess_data():
    # 이 예시에서는 CIFAR10을 사용합니다. STL-10을 사용할 경우, 적절히 대체해주세요.
    (_, _), (test_images, test_labels) = datasets.cifar10.load_data()
    test_images = test_images / 255.0
    return test_images, test_labels

def main():
    test_images, test_labels = load_and_preprocess_data()

    # 모델 로드
    model = tf.keras.models.load_model('model/my_model')

    # 모델 평가
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f'\nTest accuracy: {test_acc}, Test loss: {test_loss}')

if __name__ == '__main__':
    main()
