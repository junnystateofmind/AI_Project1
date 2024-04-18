# AI_Project1

## **서울대학교 24-1학기 인공지능 Term Project** ##



Topic: Design and Analysis Report of a CNN Model for Training STL-10

Submission: Please submit the project code (including comments) and the modeling analysis report as a single compressed ZIP file with the following naming format: Name_StudentID_pr1.zip
e.g., "박재문_2018-12967_pr1.zip"

Due Date: 5/12 23:59 PM

Report: The report should include an explanation of the designed CNN model and its architecture for training STL-10.
Additionally, provide detailed analysis of the model's performance, results, and any insights gained from working with the dataset. 

You are allowed to use chatGPT, but model modifications must be made with the goal of improving the performance of the generated model, and comparative analysis must also be included.

Dataset: STL-10: https://cs.stanford.edu/~acoates/stl10/

Grading Criteria:

Difficulty level of implementing the CNN model
Depth and quality of analysis presented in the report (maximum 3 pages)
Model's performance on the selected dataset
Note: 

While there are no programming language restrictions, Python is recommended.
PLAGIARISM WILL NOT BE TOLERATED: Please be aware that copying assignments will result in a score of 0, so ensure to submit your work thoughtfully.

# 프로젝트 개요 #
- CNN 모델을 설계하고 STL-10 데이터셋을 학습시키는 프로젝트
- Tensorflow 기반 CNN 모델 설계 및 학습

개인적으로 세운 세부 목표는 다음과 같다
- 모델 파라미터는 가능한 적으면서, 성능은 90% 이상 나오도록 설계


```bash
AI_Project1/
│
├── data/                   # 데이터셋을 저장하는 디렉토리
│   └── stl10_binary/       # STL-10 데이터셋
│
├── models/                 # 모델 아키텍처 및 학습된 모델 파일
│   ├── cnn_model.py  
│   ├── EfficientNet.py     # Transfer Learning 모델
│   └── trained_models/     # 학습된 모델 파일 저장 위치
│
├── notebooks/              # Jupyter 노트북 파일
│   └── test.ipynb
│
├── src/                    # 소스 코드
│   ├── __init__.py
│   ├── train.py               # 모델 학습 스크립트
│   └── evaluate.py            # 모델 평가 및 테스트 스크립트
│
├── utils/                  # 유틸리티 함수 및 클래스
│   ├── __init__.py
│   └── utils.py             # 유용한 함수 및 클래스 정의
│
├── requirements.txt        # 프로젝트 의존성 목록
├── .gitignore              # Git 버전 관리에서 제외할 파일 목록
└── README.md               # 프로젝트 설명, 사용 방법 등
```


# 작업 로그 # 

- 2024.04.01
  ```bash
    ==================================================================================================
    Total params: 1,397,642
    Trainable params: 1,397,642
    Non-trainable params: 0
    __________________________________________________________________________________________________
    Epoch 100/100 
    157/157 [==============================] - 20s 131ms/step - loss: 2.1908e-05 - accuracy: 1.0000 - val_loss: 3.5760 - val_accuracy: 0.6679
    35 에포크 정도에서 overfitting이 발생하는 것으로 보임
    파라미터 크기는 16.1MB으로, 더 늘려도 될 것으로 보임
    ==================================================================================================
    Epoch 59/100
    157/157 [==============================] - 28s 179ms/step - loss: 0.0039 - accuracy: 0.9998 - val_loss: 2.2383 - val_accuracy: 0.6693 
    필터 512, 1024, 2048로 늘려서 학습시켜봄
    Total params: 6,387,594
    Trainable params: 6,387,594
    Non-trainable params: 0
    학습시간만 늘어나고 성능은 크게 향상되지 않음, 필터는 이제 유의미한 성능 향상을 주지 않을 것으로 보임, 층을 하나 더 늘려보자
    __________________________________________________________________________________________________
  ```
- 2024.04.03
     ```bash
  ==================================================================================================
  2024-04-03 09:21:00.995121: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
  2024-04-03 09:21:00.995176: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
  2024-04-03 09:21:00.996631: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
  2024-04-03 09:21:01.004273: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
  To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
  2024-04-03 09:21:02.003936: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
  2024-04-03 09:21:03.727521: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:901] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
  2024-04-03 09:21:04.304892: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:901] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
  2024-04-03 09:21:04.305202: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:901] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
  2024-04-03 09:21:04.305928: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:901] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
  2024-04-03 09:21:04.306161: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:901] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
  2024-04-03 09:21:04.306342: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:901] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
  2024-04-03 09:21:04.519898: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:901] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
  2024-04-03 09:21:04.520196: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:901] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
  2024-04-03 09:21:04.520329: W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:47] Overriding orig_value setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was 0.
  2024-04-03 09:21:04.520431: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:901] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
  2024-04-03 09:21:04.520570: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1929] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 13949 MB memory:  -> device: 0, name: Tesla T4, pci bus id: 0000:00:04.0, compute capability: 7.5
  Downloading data from https://storage.googleapis.com/keras-applications/efficientnetb0_notop.h5
  16705208/16705208 [==============================] - 2s 0us/step
  Model: "model"
  _________________________________________________________________
   Layer (type)                Output Shape              Param #   
  =================================================================
   input_1 (InputLayer)        [(None, 96, 96, 3)]       0         
                                                                   
   efficientnetb0 (Functional  (None, 3, 3, 1280)        4049571   
   )                                                               
                                                                   
   global_average_pooling2d (  (None, 1280)              0         
   GlobalAveragePooling2D)                                         
                                                                   
   dense (Dense)               (None, 64)                81984     
                                                                   
   dense_1 (Dense)             (None, 10)                650       
                                                                   
  =================================================================
  Total params: 4132205 (15.76 MB)
  Trainable params: 4090182 (15.60 MB)
  Non-trainable params: 42023 (164.16 KB)
  _________________________________________________________________
  Epoch 1/70
  2024-04-03 09:22:29.387806: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:961] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape inmodel_1/efficientnetb0/block2b_drop/dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
  2024-04-03 09:22:32.455048: I external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:454] Loaded cuDNN version 8906
  2024-04-03 09:22:34.565921: I external/local_xla/xla/service/service.cc:168] XLA service 0x7e751546e5c0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
  2024-04-03 09:22:34.565969: I external/local_xla/xla/service/service.cc:176]   StreamExecutor device (0): Tesla T4, Compute Capability 7.5
  2024-04-03 09:22:34.584383: I tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc:269] disabling MLIR crash reproducer, set env var `MLIR_CRASH_REPRODUCER_DIRECTORY` to enable.
  WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
  I0000 00:00:1712136154.720043    2311 device_compiler.h:186] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
   6/79 [=>............................] - ETA: 10s - loss: 2.8951 - accuracy: 0.1823WARNING:tensorflow:Callback method `on_train_batch_end` is slow compared to the batch time (batch time: 0.1084s vs `on_train_batch_end` time: 0.1107s). Check your callbacks.
  79/79 [==============================] - ETA: 0s - loss: 1.9891 - accuracy: 0.26482024-04-03 09:23:09.292218: W external/local_tsl/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 884736000 exceeds 10% of free system memory.
  79/79 [==============================] - 66s 329ms/step - loss: 1.9891 - accuracy: 0.2648 - val_loss: 163.7273 - val_accuracy: 0.1000 - lr: 0.0100
  Epoch 2/70
  79/79 [==============================] - 22s 273ms/step - loss: 1.5922 - accuracy: 0.4050 - val_loss: 3.0064 - val_accuracy: 0.1023 - lr: 0.0100
  Epoch 3/70
  79/79 [==============================] - 22s 278ms/step - loss: 1.4360 - accuracy: 0.4566 - val_loss: 4098.2334 - val_accuracy: 0.1000 - lr: 0.0100
  Epoch 4/70
  79/79 [==============================] - 23s 293ms/step - loss: 1.4208 - accuracy: 0.4700 - val_loss: 5.3928 - val_accuracy: 0.1000 - lr: 0.0100
  Epoch 5/70
  79/79 [==============================] - ETA: 0s - loss: 1.2725 - accuracy: 0.5340/usr/local/lib/python3.10/dist-packages/keras/src/engine/training.py:3103: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
    saving_api.save_model(
  79/79 [==============================] - 22s 284ms/step - loss: 1.2725 - accuracy: 0.5340 - val_loss: 6.3674 - val_accuracy: 0.1000 - lr: 0.0100
  Epoch 6/70
  79/79 [==============================] - 24s 304ms/step - loss: 1.2794 - accuracy: 0.5394 - val_loss: 10.7344 - val_accuracy: 0.1006 - lr: 0.0100
  Epoch 7/70
  79/79 [==============================] - 24s 300ms/step - loss: 1.2059 - accuracy: 0.5726 - val_loss: 5.2114 - val_accuracy: 0.1000 - lr: 0.0100
  Epoch 8/70
  79/79 [==============================] - 24s 299ms/step - loss: 1.2288 - accuracy: 0.5544 - val_loss: 254.1098 - val_accuracy: 0.1000 - lr: 0.0100
  Epoch 9/70
  79/79 [==============================] - 23s 285ms/step - loss: 1.1326 - accuracy: 0.5906 - val_loss: 19.5576 - val_accuracy: 0.1021 - lr: 0.0100
  Epoch 10/70
  79/79 [==============================] - 23s 295ms/step - loss: 1.0629 - accuracy: 0.6148 - val_loss: 26.0437 - val_accuracy: 0.1000 - lr: 0.0100
  Epoch 11/70
  79/79 [==============================] - 24s 306ms/step - loss: 0.9847 - accuracy: 0.6512 - val_loss: 3.6911 - val_accuracy: 0.1268 - lr: 0.0090
  Epoch 12/70
  79/79 [==============================] - 22s 277ms/step - loss: 0.9612 - accuracy: 0.6558 - val_loss: 4.7139 - val_accuracy: 0.1066 - lr: 0.0082
  Epoch 13/70
  79/79 [==============================] - 23s 286ms/step - loss: 0.8782 - accuracy: 0.6798 - val_loss: 4.8076 - val_accuracy: 0.0959 - lr: 0.0074
  Epoch 14/70
  79/79 [==============================] - 22s 279ms/step - loss: 0.8398 - accuracy: 0.6972 - val_loss: 4.6202 - val_accuracy: 0.0980 - lr: 0.0067
  Epoch 15/70
  79/79 [==============================] - 24s 304ms/step - loss: 0.8064 - accuracy: 0.7096 - val_loss: 11.8305 - val_accuracy: 0.1000 - lr: 0.0061
  Epoch 16/70
  79/79 [==============================] - 23s 290ms/step - loss: 0.7552 - accuracy: 0.7298 - val_loss: 5.9780 - val_accuracy: 0.0994 - lr: 0.0055
  Epoch 17/70
  79/79 [==============================] - 23s 286ms/step - loss: 0.7354 - accuracy: 0.7356 - val_loss: 6.1529 - val_accuracy: 0.0852 - lr: 0.0050
  Epoch 18/70
  79/79 [==============================] - 23s 290ms/step - loss: 0.7032 - accuracy: 0.7550 - val_loss: 5.7651 - val_accuracy: 0.1036 - lr: 0.0045
  Epoch 19/70
  79/79 [==============================] - 23s 288ms/step - loss: 0.6663 - accuracy: 0.7682 - val_loss: 4.5242 - val_accuracy: 0.1061 - lr: 0.0041
  Epoch 20/70
  79/79 [==============================] - 22s 284ms/step - loss: 0.6155 - accuracy: 0.7830 - val_loss: 6.4192 - val_accuracy: 0.1005 - lr: 0.0037
  Epoch 21/70
  79/79 [==============================] - 23s 289ms/step - loss: 0.6004 - accuracy: 0.7890 - val_loss: 2.0696 - val_accuracy: 0.3453 - lr: 0.0033
  Epoch 22/70
  79/79 [==============================] - 22s 272ms/step - loss: 0.5848 - accuracy: 0.7954 - val_loss: 3.6572 - val_accuracy: 0.1000 - lr: 0.0030
  Epoch 23/70
  79/79 [==============================] - 23s 286ms/step - loss: 0.5429 - accuracy: 0.8086 - val_loss: 3.5907 - val_accuracy: 0.1220 - lr: 0.0027
  Epoch 24/70
  79/79 [==============================] - 21s 266ms/step - loss: 0.4944 - accuracy: 0.8200 - val_loss: 3.7090 - val_accuracy: 0.1490 - lr: 0.0025
  Epoch 25/70
  79/79 [==============================] - 23s 286ms/step - loss: 0.5268 - accuracy: 0.8108 - val_loss: 5.5080 - val_accuracy: 0.1009 - lr: 0.0022
  Epoch 26/70
  79/79 [==============================] - 23s 289ms/step - loss: 0.4932 - accuracy: 0.8264 - val_loss: 3.8166 - val_accuracy: 0.1360 - lr: 0.0020
  Epoch 27/70
  79/79 [==============================] - 23s 292ms/step - loss: 0.4960 - accuracy: 0.8196 - val_loss: 4.9507 - val_accuracy: 0.1047 - lr: 0.0018
  Epoch 28/70
  79/79 [==============================] - 21s 265ms/step - loss: 0.4701 - accuracy: 0.8292 - val_loss: 3.6343 - val_accuracy: 0.1093 - lr: 0.0017
  Epoch 29/70
  79/79 [==============================] - 23s 287ms/step - loss: 0.4610 - accuracy: 0.8334 - val_loss: 3.8194 - val_accuracy: 0.1395 - lr: 0.0015
  Epoch 30/70
  79/79 [==============================] - 23s 298ms/step - loss: 0.4153 - accuracy: 0.8546 - val_loss: 4.7150 - val_accuracy: 0.1716 - lr: 0.0014
  Epoch 31/70
  79/79 [==============================] - 22s 282ms/step - loss: 0.4072 - accuracy: 0.8582 - val_loss: 4.0003 - val_accuracy: 0.2076 - lr: 0.0012
  Epoch 32/70
  79/79 [==============================] - 22s 284ms/step - loss: 0.4042 - accuracy: 0.8624 - val_loss: 3.4753 - val_accuracy: 0.2327 - lr: 0.0011
  Epoch 33/70
  79/79 [==============================] - 21s 269ms/step - loss: 0.3714 - accuracy: 0.8708 - val_loss: 3.2573 - val_accuracy: 0.2840 - lr: 0.0010
  Epoch 34/70
  79/79 [==============================] - 22s 273ms/step - loss: 0.3828 - accuracy: 0.8614 - val_loss: 4.7952 - val_accuracy: 0.1639 - lr: 9.0718e-04
  Epoch 35/70
  79/79 [==============================] - 22s 275ms/step - loss: 0.3726 - accuracy: 0.8662 - val_loss: 4.0841 - val_accuracy: 0.2260 - lr: 8.2085e-04
  Epoch 36/70
  79/79 [==============================] - 22s 273ms/step - loss: 0.3494 - accuracy: 0.8774 - val_loss: 4.8084 - val_accuracy: 0.2537 - lr: 7.4273e-04
  Epoch 37/70
  79/79 [==============================] - 23s 296ms/step - loss: 0.3502 - accuracy: 0.8726 - val_loss: 3.7019 - val_accuracy: 0.2186 - lr: 6.7205e-04
  Epoch 38/70
  79/79 [==============================] - 22s 285ms/step - loss: 0.3354 - accuracy: 0.8776 - val_loss: 2.7407 - val_accuracy: 0.2469 - lr: 6.0810e-04
  Epoch 39/70
  79/79 [==============================] - 21s 270ms/step - loss: 0.3401 - accuracy: 0.8762 - val_loss: 2.8737 - val_accuracy: 0.3313 - lr: 5.5023e-04
  Epoch 40/70
  79/79 [==============================] - 22s 282ms/step - loss: 0.3278 - accuracy: 0.8818 - val_loss: 2.4983 - val_accuracy: 0.3754 - lr: 4.9787e-04
  Epoch 41/70
  79/79 [==============================] - 23s 296ms/step - loss: 0.3185 - accuracy: 0.8852 - val_loss: 3.2969 - val_accuracy: 0.2569 - lr: 4.5049e-04
  Epoch 42/70
  79/79 [==============================] - 22s 285ms/step - loss: 0.3075 - accuracy: 0.8906 - val_loss: 2.6593 - val_accuracy: 0.3351 - lr: 4.0762e-04
  Epoch 43/70
  79/79 [==============================] - 23s 294ms/step - loss: 0.3000 - accuracy: 0.8896 - val_loss: 1.8643 - val_accuracy: 0.5269 - lr: 3.6883e-04
  Epoch 44/70
  79/79 [==============================] - 21s 262ms/step - loss: 0.3058 - accuracy: 0.8976 - val_loss: 1.6107 - val_accuracy: 0.6089 - lr: 3.3373e-04
  Epoch 45/70
  79/79 [==============================] - 23s 296ms/step - loss: 0.2904 - accuracy: 0.8976 - val_loss: 4.3164 - val_accuracy: 0.3113 - lr: 3.0197e-04
  Epoch 46/70
  79/79 [==============================] - 23s 291ms/step - loss: 0.2971 - accuracy: 0.8938 - val_loss: 1.5010 - val_accuracy: 0.6329 - lr: 2.7324e-04
  Epoch 47/70
  79/79 [==============================] - 23s 287ms/step - loss: 0.2870 - accuracy: 0.8994 - val_loss: 2.4113 - val_accuracy: 0.4686 - lr: 2.4723e-04
  Epoch 48/70
  79/79 [==============================] - 23s 293ms/step - loss: 0.2817 - accuracy: 0.8998 - val_loss: 3.7471 - val_accuracy: 0.2834 - lr: 2.2371e-04
  Epoch 49/70
  79/79 [==============================] - 23s 291ms/step - loss: 0.2678 - accuracy: 0.9050 - val_loss: 2.5517 - val_accuracy: 0.4190 - lr: 2.0242e-04
  Epoch 50/70
  79/79 [==============================] - 23s 295ms/step - loss: 0.2584 - accuracy: 0.9064 - val_loss: 2.4194 - val_accuracy: 0.5048 - lr: 1.8316e-04
  Epoch 51/70
  79/79 [==============================] - 22s 273ms/step - loss: 0.2734 - accuracy: 0.9070 - val_loss: 3.6961 - val_accuracy: 0.2176 - lr: 1.6573e-04
  Epoch 52/70
  79/79 [==============================] - 22s 283ms/step - loss: 0.2527 - accuracy: 0.9082 - val_loss: 3.2802 - val_accuracy: 0.3765 - lr: 1.4996e-04
  Epoch 53/70
  79/79 [==============================] - 21s 269ms/step - loss: 0.2521 - accuracy: 0.9076 - val_loss: 1.5058 - val_accuracy: 0.6291 - lr: 1.3569e-04
  Epoch 54/70
  79/79 [==============================] - 21s 268ms/step - loss: 0.2484 - accuracy: 0.9092 - val_loss: 2.6792 - val_accuracy: 0.4636 - lr: 1.2277e-04
  Epoch 55/70
  79/79 [==============================] - 24s 308ms/step - loss: 0.2605 - accuracy: 0.9068 - val_loss: 2.2701 - val_accuracy: 0.4996 - lr: 1.1109e-04
  Epoch 56/70
  79/79 [==============================] - 21s 269ms/step - loss: 0.2533 - accuracy: 0.9128 - val_loss: 1.9821 - val_accuracy: 0.5694 - lr: 1.0052e-04
  Epoch 57/70
  79/79 [==============================] - 23s 296ms/step - loss: 0.2544 - accuracy: 0.9078 - val_loss: 0.9302 - val_accuracy: 0.7459 - lr: 9.0953e-05
  Epoch 58/70
  79/79 [==============================] - 22s 283ms/step - loss: 0.2555 - accuracy: 0.9078 - val_loss: 1.0198 - val_accuracy: 0.7274 - lr: 8.2297e-05
  Epoch 59/70
  79/79 [==============================] - 21s 263ms/step - loss: 0.2546 - accuracy: 0.9120 - val_loss: 1.5777 - val_accuracy: 0.5939 - lr: 7.4466e-05
  Epoch 60/70
  79/79 [==============================] - 23s 288ms/step - loss: 0.2341 - accuracy: 0.9156 - val_loss: 0.7730 - val_accuracy: 0.7751 - lr: 6.7379e-05
  Epoch 61/70
  79/79 [==============================] - 23s 286ms/step - loss: 0.2556 - accuracy: 0.9082 - val_loss: 1.1910 - val_accuracy: 0.6805 - lr: 6.0967e-05
  Epoch 62/70
  79/79 [==============================] - 22s 279ms/step - loss: 0.2491 - accuracy: 0.9108 - val_loss: 1.9489 - val_accuracy: 0.5356 - lr: 5.5166e-05
  Epoch 63/70
  79/79 [==============================] - 22s 284ms/step - loss: 0.2564 - accuracy: 0.9120 - val_loss: 0.6168 - val_accuracy: 0.8186 - lr: 4.9916e-05
  Epoch 64/70
  79/79 [==============================] - 22s 281ms/step - loss: 0.2624 - accuracy: 0.9036 - val_loss: 0.7796 - val_accuracy: 0.7774 - lr: 4.5166e-05
  Epoch 65/70
  79/79 [==============================] - 22s 274ms/step - loss: 0.2510 - accuracy: 0.9118 - val_loss: 1.0385 - val_accuracy: 0.7175 - lr: 4.0868e-05
  Epoch 66/70
  79/79 [==============================] - 22s 285ms/step - loss: 0.2413 - accuracy: 0.9132 - val_loss: 0.7416 - val_accuracy: 0.7910 - lr: 3.6979e-05
  Epoch 67/70
  79/79 [==============================] - 21s 270ms/step - loss: 0.2501 - accuracy: 0.9134 - val_loss: 0.8353 - val_accuracy: 0.7681 - lr: 3.3460e-05
  Epoch 68/70
  79/79 [==============================] - 22s 282ms/step - loss: 0.2461 - accuracy: 0.9148 - val_loss: 0.7392 - val_accuracy: 0.7928 - lr: 3.0275e-05
  Epoch 69/70
  79/79 [==============================] - 21s 269ms/step - loss: 0.2358 - accuracy: 0.9142 - val_loss: 0.6261 - val_accuracy: 0.8160 - lr: 2.7394e-05
  Epoch 70/70
  79/79 [==============================] - 22s 284ms/step - loss: 0.2482 - accuracy: 0.9142 - val_loss: 0.6617 - val_accuracy: 0.8062 - lr: 2.4787e-05
  __________________________________________________________________________________________________
  val_accuracy가 갑자기 확 오르는 경우가 생김 흠...
  Transfer Learning에서는 Valid Accuracy가 느리게 수렴해 이에 따라 스케줄러를 다르게 적용할 필요가 있어보임
  ```
- 2024.04.04
     ```bash
  2024-04-04 09:26:19.454781: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
  2024-04-04 09:26:19.454831: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
  2024-04-04 09:26:19.456308: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
  2024-04-04 09:26:19.464512: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
  To enable the following instructions: AVX2 AVX512F FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
  2024-04-04 09:26:20.546734: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
  2024-04-04 09:26:23.258439: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:901] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
  2024-04-04 09:26:23.310472: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:901] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
  2024-04-04 09:26:23.310788: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:901] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
  2024-04-04 09:26:23.311600: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:901] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
  2024-04-04 09:26:23.311826: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:901] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
  2024-04-04 09:26:23.311995: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:901] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
  2024-04-04 09:26:23.408097: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:901] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
  2024-04-04 09:26:23.408366: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:901] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
  2024-04-04 09:26:23.408516: W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:47] Overriding orig_value setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was 0.
  2024-04-04 09:26:23.408621: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:901] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
  2024-04-04 09:26:23.408762: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1929] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 13949 MB memory:  -> device: 0, name: Tesla T4, pci bus id: 0000:00:04.0, compute capability: 7.5
  2024-04-04 09:26:24.562122: W external/local_tsl/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 552960000 exceeds 10% of free system memory.
  Epoch 1/100
  2024-04-04 09:27:00.493684: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:961] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape inmodel/efficientnetb3/block1b_drop/dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
  2024-04-04 09:27:05.360175: I external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:454] Loaded cuDNN version 8906
  2024-04-04 09:27:08.502092: I external/local_xla/xla/service/service.cc:168] XLA service 0x7e5fd1c63fc0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
  2024-04-04 09:27:08.502141: I external/local_xla/xla/service/service.cc:176]   StreamExecutor device (0): Tesla T4, Compute Capability 7.5
  2024-04-04 09:27:08.512583: I tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc:269] disabling MLIR crash reproducer, set env var `MLIR_CRASH_REPRODUCER_DIRECTORY` to enable.
  WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
  I0000 00:00:1712222828.684226   12823 device_compiler.h:186] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
  20/20 [==============================] - ETA: 0s - loss: 2.1325 - accuracy: 0.29142024-04-04 09:27:54.315320: W external/local_tsl/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 884736000 exceeds 10% of free system memory.
  20/20 [==============================] - 100s 2s/step - loss: 2.1325 - accuracy: 0.2914 - val_loss: 32907577344.0000 - val_accuracy: 0.1000 - lr: 0.0100
  Epoch 2/100
  20/20 [==============================] - 32s 2s/step - loss: 1.4523 - accuracy: 0.4588 - val_loss: 33173753856.0000 - val_accuracy: 0.1000 - lr: 0.0100
  Epoch 3/100
  20/20 [==============================] - 35s 2s/step - loss: 1.3025 - accuracy: 0.5130 - val_loss: 1385018.8750 - val_accuracy: 0.1000 - lr: 0.0100
  Epoch 4/100
  20/20 [==============================] - 32s 2s/step - loss: 1.1304 - accuracy: 0.5818 - val_loss: 2212.1543 - val_accuracy: 0.1000 - lr: 0.0100
  Epoch 5/100
  20/20 [==============================] - 35s 2s/step - loss: 1.0730 - accuracy: 0.6142 - val_loss: 128.8517 - val_accuracy: 0.1000 - lr: 0.0100
  Epoch 6/100
  20/20 [==============================] - 35s 2s/step - loss: 0.9981 - accuracy: 0.6400 - val_loss: 332.0110 - val_accuracy: 0.1000 - lr: 0.0100
  Epoch 7/100
  20/20 [==============================] - 34s 2s/step - loss: 0.8718 - accuracy: 0.6888 - val_loss: 592.0634 - val_accuracy: 0.1000 - lr: 0.0100
  Epoch 8/100
  20/20 [==============================] - 36s 2s/step - loss: 0.8196 - accuracy: 0.7038 - val_loss: 4675.1675 - val_accuracy: 0.1000 - lr: 0.0100
  Epoch 9/100
  20/20 [==============================] - 32s 2s/step - loss: 0.7715 - accuracy: 0.7278 - val_loss: 13.7131 - val_accuracy: 0.1000 - lr: 0.0100
  Epoch 10/100
  20/20 [==============================] - ETA: 0s - loss: 0.7300 - accuracy: 0.7486/usr/local/lib/python3.10/dist-packages/keras/src/engine/training.py:3103: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
    saving_api.save_model(
  20/20 [==============================] - 44s 2s/step - loss: 0.7300 - accuracy: 0.7486 - val_loss: 8.5190 - val_accuracy: 0.1000 - lr: 0.0100
  Epoch 11/100
  20/20 [==============================] - 35s 2s/step - loss: 0.7107 - accuracy: 0.7446 - val_loss: 10.4412 - val_accuracy: 0.1000 - lr: 0.0090
  Epoch 12/100
  20/20 [==============================] - 31s 2s/step - loss: 0.6621 - accuracy: 0.7734 - val_loss: 10.2468 - val_accuracy: 0.1000 - lr: 0.0082
  Epoch 13/100
  20/20 [==============================] - 30s 2s/step - loss: 0.6282 - accuracy: 0.7764 - val_loss: 3.2888 - val_accuracy: 0.1030 - lr: 0.0074
  Epoch 14/100
  20/20 [==============================] - 36s 2s/step - loss: 0.5657 - accuracy: 0.8062 - val_loss: 4.7193 - val_accuracy: 0.0994 - lr: 0.0067
  Epoch 15/100
  20/20 [==============================] - 34s 2s/step - loss: 0.4941 - accuracy: 0.8250 - val_loss: 3.9463 - val_accuracy: 0.0994 - lr: 0.0061
  Epoch 16/100
  20/20 [==============================] - 36s 2s/step - loss: 0.4976 - accuracy: 0.8212 - val_loss: 6.2096 - val_accuracy: 0.0911 - lr: 0.0055
  Epoch 17/100
  20/20 [==============================] - 35s 2s/step - loss: 0.4263 - accuracy: 0.8514 - val_loss: 5.7741 - val_accuracy: 0.1000 - lr: 0.0050
  Epoch 18/100
  20/20 [==============================] - 30s 2s/step - loss: 0.4075 - accuracy: 0.8598 - val_loss: 3.7077 - val_accuracy: 0.1009 - lr: 0.0045
  Epoch 19/100
  20/20 [==============================] - 35s 2s/step - loss: 0.3843 - accuracy: 0.8648 - val_loss: 10.4970 - val_accuracy: 0.1000 - lr: 0.0041
  Epoch 20/100
  20/20 [==============================] - 34s 2s/step - loss: 0.3467 - accuracy: 0.8782 - val_loss: 5.3337 - val_accuracy: 0.1047 - lr: 0.0037
  Epoch 21/100
  20/20 [==============================] - 35s 2s/step - loss: 0.3210 - accuracy: 0.8934 - val_loss: 4.5549 - val_accuracy: 0.1220 - lr: 0.0033
  Epoch 22/100
  20/20 [==============================] - 34s 2s/step - loss: 0.2747 - accuracy: 0.9052 - val_loss: 3.9679 - val_accuracy: 0.1015 - lr: 0.0030
  Epoch 23/100
  20/20 [==============================] - 35s 2s/step - loss: 0.2710 - accuracy: 0.9028 - val_loss: 3.1206 - val_accuracy: 0.1187 - lr: 0.0027
  Epoch 24/100
  20/20 [==============================] - 33s 2s/step - loss: 0.2503 - accuracy: 0.9106 - val_loss: 3.5655 - val_accuracy: 0.1002 - lr: 0.0025
  Epoch 25/100
  20/20 [==============================] - 34s 2s/step - loss: 0.2305 - accuracy: 0.9168 - val_loss: 4.0430 - val_accuracy: 0.1000 - lr: 0.0022
  Epoch 26/100
  20/20 [==============================] - 31s 2s/step - loss: 0.2221 - accuracy: 0.9252 - val_loss: 4.0093 - val_accuracy: 0.0925 - lr: 0.0020
  Epoch 27/100
  20/20 [==============================] - 30s 2s/step - loss: 0.2160 - accuracy: 0.9250 - val_loss: 3.8117 - val_accuracy: 0.1377 - lr: 0.0018
  Epoch 28/100
  20/20 [==============================] - 35s 2s/step - loss: 0.1814 - accuracy: 0.9362 - val_loss: 2.6674 - val_accuracy: 0.3511 - lr: 0.0017
  Epoch 29/100
  20/20 [==============================] - 31s 2s/step - loss: 0.1724 - accuracy: 0.9414 - val_loss: 2.8216 - val_accuracy: 0.2716 - lr: 0.0015
  Epoch 30/100
  20/20 [==============================] - 34s 2s/step - loss: 0.1628 - accuracy: 0.9432 - val_loss: 4.2656 - val_accuracy: 0.2084 - lr: 0.0014
  Epoch 31/100
  20/20 [==============================] - 34s 2s/step - loss: 0.1673 - accuracy: 0.9424 - val_loss: 4.6168 - val_accuracy: 0.1721 - lr: 0.0012
  Epoch 32/100
  20/20 [==============================] - 34s 2s/step - loss: 0.1357 - accuracy: 0.9522 - val_loss: 4.9460 - val_accuracy: 0.2011 - lr: 0.0011
  Epoch 33/100
  20/20 [==============================] - 35s 2s/step - loss: 0.1299 - accuracy: 0.9556 - val_loss: 2.7148 - val_accuracy: 0.3754 - lr: 0.0010
  Epoch 34/100
  20/20 [==============================] - 35s 2s/step - loss: 0.1114 - accuracy: 0.9612 - val_loss: 2.3643 - val_accuracy: 0.4570 - lr: 9.0718e-04
  Epoch 35/100
  20/20 [==============================] - 34s 2s/step - loss: 0.1131 - accuracy: 0.9626 - val_loss: 2.8249 - val_accuracy: 0.3660 - lr: 8.2085e-04
  Epoch 36/100
  20/20 [==============================] - 30s 2s/step - loss: 0.1128 - accuracy: 0.9586 - val_loss: 2.2664 - val_accuracy: 0.4894 - lr: 7.4273e-04
  Epoch 37/100
  20/20 [==============================] - 30s 2s/step - loss: 0.1002 - accuracy: 0.9658 - val_loss: 7.0591 - val_accuracy: 0.1011 - lr: 6.7205e-04
  Epoch 38/100
  20/20 [==============================] - 33s 2s/step - loss: 0.1057 - accuracy: 0.9604 - val_loss: 7.6825 - val_accuracy: 0.1011 - lr: 6.0810e-04
  Epoch 39/100
  20/20 [==============================] - 34s 2s/step - loss: 0.0962 - accuracy: 0.9684 - val_loss: 7.4776 - val_accuracy: 0.1053 - lr: 5.5023e-04
  Epoch 40/100
  20/20 [==============================] - 32s 2s/step - loss: 0.0895 - accuracy: 0.9702 - val_loss: 3.0392 - val_accuracy: 0.3246 - lr: 4.9787e-04
  Epoch 41/100
  20/20 [==============================] - 32s 2s/step - loss: 0.0863 - accuracy: 0.9714 - val_loss: 4.9480 - val_accuracy: 0.2246 - lr: 4.5049e-04
  Epoch 42/100
  20/20 [==============================] - 34s 2s/step - loss: 0.0831 - accuracy: 0.9734 - val_loss: 1.8396 - val_accuracy: 0.6141 - lr: 4.0762e-04
  Epoch 43/100
  20/20 [==============================] - 35s 2s/step - loss: 0.0824 - accuracy: 0.9722 - val_loss: 6.9307 - val_accuracy: 0.1177 - lr: 3.6883e-04
  Epoch 44/100
  20/20 [==============================] - 30s 2s/step - loss: 0.0822 - accuracy: 0.9710 - val_loss: 2.2205 - val_accuracy: 0.5311 - lr: 3.3373e-04
  Epoch 45/100
  20/20 [==============================] - 35s 2s/step - loss: 0.0762 - accuracy: 0.9724 - val_loss: 2.7562 - val_accuracy: 0.5228 - lr: 3.0197e-04
  Epoch 46/100
  20/20 [==============================] - 34s 2s/step - loss: 0.0740 - accuracy: 0.9770 - val_loss: 2.5422 - val_accuracy: 0.5514 - lr: 2.7324e-04
  Epoch 47/100
  20/20 [==============================] - 32s 2s/step - loss: 0.0707 - accuracy: 0.9750 - val_loss: 3.7038 - val_accuracy: 0.4079 - lr: 2.4723e-04
  Epoch 48/100
  20/20 [==============================] - 32s 2s/step - loss: 0.0697 - accuracy: 0.9742 - val_loss: 4.1753 - val_accuracy: 0.3085 - lr: 2.2371e-04
  Epoch 49/100
  20/20 [==============================] - 32s 2s/step - loss: 0.0675 - accuracy: 0.9766 - val_loss: 1.6498 - val_accuracy: 0.6790 - lr: 2.0242e-04
  Epoch 50/100
  20/20 [==============================] - 34s 2s/step - loss: 0.0633 - accuracy: 0.9768 - val_loss: 0.9636 - val_accuracy: 0.7716 - lr: 1.8316e-04
  Epoch 51/100
  20/20 [==============================] - 31s 2s/step - loss: 0.0721 - accuracy: 0.9772 - val_loss: 0.7062 - val_accuracy: 0.8366 - lr: 1.6573e-04
  Epoch 52/100
  20/20 [==============================] - 30s 2s/step - loss: 0.0601 - accuracy: 0.9816 - val_loss: 2.5219 - val_accuracy: 0.5257 - lr: 1.4996e-04
  Epoch 53/100
  20/20 [==============================] - 36s 2s/step - loss: 0.0695 - accuracy: 0.9774 - val_loss: 2.1198 - val_accuracy: 0.5853 - lr: 1.3569e-04
  Epoch 54/100
  20/20 [==============================] - 30s 2s/step - loss: 0.0723 - accuracy: 0.9756 - val_loss: 1.4386 - val_accuracy: 0.6914 - lr: 1.2277e-04
  Epoch 55/100
  20/20 [==============================] - 35s 2s/step - loss: 0.0652 - accuracy: 0.9788 - val_loss: 1.7573 - val_accuracy: 0.6384 - lr: 1.1109e-04
  Epoch 56/100
  20/20 [==============================] - 37s 2s/step - loss: 0.0614 - accuracy: 0.9774 - val_loss: 1.4232 - val_accuracy: 0.6931 - lr: 1.0052e-04
  Epoch 57/100
  20/20 [==============================] - 30s 2s/step - loss: 0.0575 - accuracy: 0.9802 - val_loss: 1.1344 - val_accuracy: 0.7462 - lr: 9.0953e-05
  Epoch 58/100
  20/20 [==============================] - 35s 2s/step - loss: 0.0594 - accuracy: 0.9818 - val_loss: 1.9138 - val_accuracy: 0.6246 - lr: 8.2297e-05
  Epoch 59/100
  20/20 [==============================] - 30s 2s/step - loss: 0.0591 - accuracy: 0.9816 - val_loss: 1.9777 - val_accuracy: 0.6130 - lr: 7.4466e-05
  Epoch 60/100
  20/20 [==============================] - 34s 2s/step - loss: 0.0536 - accuracy: 0.9818 - val_loss: 1.1586 - val_accuracy: 0.7536 - lr: 6.7379e-05
  Epoch 61/100
  20/20 [==============================] - 31s 2s/step - loss: 0.0549 - accuracy: 0.9812 - val_loss: 1.0249 - val_accuracy: 0.7793 - lr: 6.0967e-05
  Epoch 62/100
  20/20 [==============================] - 35s 2s/step - loss: 0.0602 - accuracy: 0.9776 - val_loss: 1.0810 - val_accuracy: 0.7713 - lr: 5.5166e-05
  Epoch 63/100
  20/20 [==============================] - 31s 2s/step - loss: 0.0644 - accuracy: 0.9792 - val_loss: 0.8483 - val_accuracy: 0.8126 - lr: 4.9916e-05
  Epoch 64/100
  20/20 [==============================] - 36s 2s/step - loss: 0.0591 - accuracy: 0.9796 - val_loss: 0.9112 - val_accuracy: 0.7984 - lr: 4.5166e-05
  Epoch 65/100
  20/20 [==============================] - 32s 2s/step - loss: 0.0531 - accuracy: 0.9814 - val_loss: 0.8411 - val_accuracy: 0.8106 - lr: 4.0868e-05
  Epoch 66/100
  20/20 [==============================] - 34s 2s/step - loss: 0.0526 - accuracy: 0.9828 - val_loss: 0.8044 - val_accuracy: 0.8183 - lr: 3.6979e-05
  Epoch 67/100
  20/20 [==============================] - 36s 2s/step - loss: 0.0534 - accuracy: 0.9808 - val_loss: 0.8903 - val_accuracy: 0.8054 - lr: 3.3460e-05
  Epoch 68/100
  20/20 [==============================] - 30s 2s/step - loss: 0.0589 - accuracy: 0.9790 - val_loss: 0.6818 - val_accuracy: 0.8420 - lr: 3.0275e-05
  Epoch 69/100
  20/20 [==============================] - 35s 2s/step - loss: 0.0499 - accuracy: 0.9806 - val_loss: 0.6848 - val_accuracy: 0.8430 - lr: 2.7394e-05
  Epoch 70/100
  20/20 [==============================] - 32s 2s/step - loss: 0.0572 - accuracy: 0.9780 - val_loss: 0.6718 - val_accuracy: 0.8461 - lr: 2.4787e-05
  Epoch 71/100
  20/20 [==============================] - 31s 2s/step - loss: 0.0534 - accuracy: 0.9804 - val_loss: 0.6531 - val_accuracy: 0.8489 - lr: 2.2429e-05
  Epoch 72/100
  20/20 [==============================] - 32s 2s/step - loss: 0.0532 - accuracy: 0.9816 - val_loss: 0.6417 - val_accuracy: 0.8518 - lr: 2.0294e-05
  Epoch 73/100
  20/20 [==============================] - 34s 2s/step - loss: 0.0550 - accuracy: 0.9810 - val_loss: 0.6352 - val_accuracy: 0.8525 - lr: 1.8363e-05
  Epoch 74/100
  20/20 [==============================] - 34s 2s/step - loss: 0.0587 - accuracy: 0.9790 - val_loss: 0.6598 - val_accuracy: 0.8478 - lr: 1.6616e-05
  Epoch 75/100
  20/20 [==============================] - 35s 2s/step - loss: 0.0537 - accuracy: 0.9830 - val_loss: 0.6545 - val_accuracy: 0.8496 - lr: 1.5034e-05
  Epoch 76/100
  20/20 [==============================] - 34s 2s/step - loss: 0.0574 - accuracy: 0.9810 - val_loss: 0.6521 - val_accuracy: 0.8508 - lr: 1.3604e-05
  Epoch 77/100
  20/20 [==============================] - 31s 2s/step - loss: 0.0696 - accuracy: 0.9774 - val_loss: 0.6425 - val_accuracy: 0.8516 - lr: 1.2309e-05
  Epoch 78/100
  20/20 [==============================] - 34s 2s/step - loss: 0.0527 - accuracy: 0.9812 - val_loss: 0.6556 - val_accuracy: 0.8508 - lr: 1.1138e-05
  Epoch 79/100
  20/20 [==============================] - 32s 2s/step - loss: 0.0615 - accuracy: 0.9782 - val_loss: 0.6490 - val_accuracy: 0.8522 - lr: 1.0078e-05
  Epoch 80/100
  20/20 [==============================] - 33s 2s/step - loss: 0.0614 - accuracy: 0.9792 - val_loss: 0.6466 - val_accuracy: 0.8509 - lr: 9.1188e-06
  Epoch 81/100
  20/20 [==============================] - 35s 2s/step - loss: 0.0587 - accuracy: 0.9812 - val_loss: 0.6269 - val_accuracy: 0.8535 - lr: 8.2510e-06
  Epoch 82/100
  20/20 [==============================] - 34s 2s/step - loss: 0.0593 - accuracy: 0.9774 - val_loss: 0.6273 - val_accuracy: 0.8524 - lr: 7.4658e-06
  Epoch 83/100
  20/20 [==============================] - 30s 2s/step - loss: 0.0582 - accuracy: 0.9816 - val_loss: 0.6163 - val_accuracy: 0.8575 - lr: 6.7554e-06
  Epoch 84/100
  20/20 [==============================] - 33s 2s/step - loss: 0.0531 - accuracy: 0.9830 - val_loss: 0.6112 - val_accuracy: 0.8562 - lr: 6.1125e-06
  Epoch 85/100
  20/20 [==============================] - 35s 2s/step - loss: 0.0600 - accuracy: 0.9802 - val_loss: 0.6151 - val_accuracy: 0.8555 - lr: 5.5308e-06
  Epoch 86/100
  20/20 [==============================] - 35s 2s/step - loss: 0.0599 - accuracy: 0.9808 - val_loss: 0.6192 - val_accuracy: 0.8543 - lr: 5.0045e-06
  Epoch 87/100
  20/20 [==============================] - 30s 2s/step - loss: 0.0621 - accuracy: 0.9792 - val_loss: 0.6215 - val_accuracy: 0.8528 - lr: 4.5283e-06
  Epoch 88/100
  20/20 [==============================] - 35s 2s/step - loss: 0.0605 - accuracy: 0.9806 - val_loss: 0.6204 - val_accuracy: 0.8534 - lr: 4.0973e-06
  Epoch 89/100
  20/20 [==============================] - 30s 2s/step - loss: 0.0589 - accuracy: 0.9786 - val_loss: 0.6284 - val_accuracy: 0.8528 - lr: 3.7074e-06
  Epoch 90/100
  20/20 [==============================] - 37s 2s/step - loss: 0.0506 - accuracy: 0.9832 - val_loss: 0.6282 - val_accuracy: 0.8529 - lr: 3.3546e-06
  Epoch 91/100
  20/20 [==============================] - 33s 2s/step - loss: 0.0619 - accuracy: 0.9784 - val_loss: 0.6245 - val_accuracy: 0.8539 - lr: 3.0354e-06
  Epoch 92/100
  20/20 [==============================] - 34s 2s/step - loss: 0.0595 - accuracy: 0.9788 - val_loss: 0.6204 - val_accuracy: 0.8558 - lr: 2.7465e-06
  Epoch 93/100
  20/20 [==============================] - 30s 2s/step - loss: 0.0535 - accuracy: 0.9822 - val_loss: 0.6160 - val_accuracy: 0.8564 - lr: 2.4852e-06
  Epoch 94/100
  20/20 [==============================] - 30s 2s/step - loss: 0.0524 - accuracy: 0.9804 - val_loss: 0.6167 - val_accuracy: 0.8560 - lr: 2.2487e-06
  Epoch 95/100
  20/20 [==============================] - 32s 2s/step - loss: 0.0512 - accuracy: 0.9820 - val_loss: 0.6168 - val_accuracy: 0.8561 - lr: 2.0347e-06
  Epoch 96/100
  20/20 [==============================] - 31s 2s/step - loss: 0.0516 - accuracy: 0.9826 - val_loss: 0.6214 - val_accuracy: 0.8551 - lr: 1.8410e-06
  Epoch 97/100
  20/20 [==============================] - 30s 2s/step - loss: 0.0449 - accuracy: 0.9854 - val_loss: 0.6215 - val_accuracy: 0.8547 - lr: 1.6659e-06
  Epoch 98/100
  20/20 [==============================] - 35s 2s/step - loss: 0.0530 - accuracy: 0.9812 - val_loss: 0.6231 - val_accuracy: 0.8547 - lr: 1.5073e-06
  Epoch 99/100
  20/20 [==============================] - 35s 2s/step - loss: 0.0564 - accuracy: 0.9780 - val_loss: 0.6239 - val_accuracy: 0.8554 - lr: 1.3639e-06
  Epoch 100/100
  20/20 [==============================] - 37s 2s/step - loss: 0.0596 - accuracy: 0.9792 - val_loss: 0.6223 - val_accuracy: 0.8551 - lr: 1.2341e-06   

    ==================================================================================================
```     
# How to run in Colab #

드라이브 마운트
```bash
from google.colab import drive
drive.mount('/content/drive')
```

프로젝트 디렉토리 이동 및 git clone
```bash
%cd /content/drive/MyDrive/AI_Project1/
!git clone https://github.com/junnystateofmind/AI_Project1.git
```
git pull
```bash
!git checkout -- src/__pycache__/train.cpython-310.pyc # remove cache file
!git pull
```
training data 다운로드
```bash
# loading training data
from torchvision import datasets
import torchvision.transforms as transforms
import os

path2data = '/content/drive/MyDrive/AI_Project1/data'

# if not exists the path, make the path
if not os.path.exists(path2data):
    os.mkdir(path2data)

data_transformer = transforms.Compose([transforms.ToTensor()])
train_ds = datasets.STL10(path2data, split='train', download='True', transform=data_transformer)

print(train_ds.data.shape)
```

모델 학습
```bash
!python -m src.train --model=(CNN, EfficientNetB0, EfficientNetB4) --epochs=50 --batch_size=64 --lr=0.01  --start_epoch=0
```

unfreeze 학습
```bash
# unfreeze
!python -m src.unfreeze_train --model=EfficientNetB4 --epochs=50 --start_epoch=70 --num_unfreeze_layer=5 --learning_rate=0.001 --batch_size=128
```

모델 평가
```bash
!python -m src.evaluate --model=(CNN, EfficientNet)
```

텐서보드 실행
```bash
%load_ext tensorboard
%tensorboard --logdir models/logs/fit
```
