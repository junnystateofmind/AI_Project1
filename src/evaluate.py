import tensorflow as tf
from tensorflow.keras import datasets
from .train import load_and_preprocess_data
import os

# 데이터 로드 및 전처리

def main():
    test_images, test_labels = load_and_preprocess_data()

    # 모델 로드
    model = tf.keras.models.load_model('model/cnn.h5')

    # TensorBoard 로그 디렉토리 설정
    log_dir = os.path.join('logs', 'evaluate')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    # 모델 평가
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2, callbacks=[tensorboard_callback])
    print(f'\nTest accuracy: {test_acc}, Test loss: {test_loss}')

if __name__ == '__main__':
    main()
