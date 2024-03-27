from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score
import logging

def setup_logging(log_file='training.log'):
    logging.basicConfig(filename=log_file, level=logging.INFO)

def log_message(message):
    logging.info(message)

def create_data_augmentation_generator(): # 데이터 증강 생성기
    return ImageDataGenerator(
        rotation_range=20, # 이미지 회전
        width_shift_range=0.2, # 가로 이동
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

def calculate_performance_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    return accuracy, precision, recall

