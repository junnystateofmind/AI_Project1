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
│   ├── cnn_model.py        # CNN 모델 아키텍처 정의
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
!python -m src.train --model=(CNN, EfficientNet) --epochs=50 --batch_size=64 --lr=0.01
```

텐서보드 실행
```bash
%load_ext tensorboard
%tensorboard --logdir models/logs/fit
```
