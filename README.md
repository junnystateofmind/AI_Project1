# AI_Project1

**서울대학교 24-1학기 인공지능 Term Project**



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


```bash
AI_Project1/
│
├── data/                   # 데이터셋을 저장하는 디렉토리
│   ├── raw/                # 원본 STL-10 데이터셋
│   └── processed/          # 전처리된 데이터셋
│
├── models/                 # 모델 아키텍처 및 학습된 모델 파일
│   ├── cnn_model.py        # CNN 모델 아키텍처 정의
│   └── trained_models/     # 학습된 모델 파일 저장 위치
│
├── notebooks/              # Jupyter 노트북 파일 (탐색적 데이터 분석, 시각화 등)
│   └── EDA_and_Visualization.ipynb
│
├── src/                    # 소스 코드
│   ├── __init__.py
│   ├── data_preprocessing.py  # 데이터 전처리 관련 코드
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
