import tensorflow as tf
import os
import numpy as np

# Set the path to store the data
path2data = '/root/AI_Project1/data/'

# keras로 STL10 데이터셋 저장

# 데이터셋 로드
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# class_name이랑




