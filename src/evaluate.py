import tensorflow as tf
from tensorflow.keras import datasets
from .train import load_and_preprocess_data
import os
import argparse

# 데이터 로드 및 전처리

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='cnn', help='Model to evaluate (cnn or efficientnet)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to evaluate the model')

    test_images, test_labels = load_and_preprocess_data()

    args = parser.parse_args()
    # 모델 로드
    if args.model == 'CNN':
        model = tf.keras.models.load_model('models/trained_models/cnn_epoch_' + str(args.epochs) + '.h5')
    elif args.model == 'EfficientNet':
        model = tf.keras.models.load_model('models/trained_models/EfficientNet_epoch_' + str(args.epochs) + '.h5')
    else:
        raise ValueError('Unknown model type: {}'.format(args.model))

    # TensorBoard 로그 디렉토리 설정
    log_dir = os.path.join('logs', 'evaluate')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    # 모델 평가
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2, callbacks=[tensorboard_callback])
    print(f'\nTest accuracy: {test_acc}, Test loss: {test_loss}')

if __name__ == '__main__':
    main()
