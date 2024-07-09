# 인삼 품질 판독기

## 1. 시연

<img src="https://github.com/bloodmage1/Insam_Decipher/blob/main/Demonstration/play_video.gif"/>

인삼의 품질을 측정하는 모습을 간략히 나타낸 영상입니다.

<img src="https://github.com/bloodmage1/Insam_Decipher/blob/main/Demonstration/insam_decipher_homescreen.png"/>

어플을 실행시켰을 때, 실행되는 인상 품질 판독기의 첫화면입니다.

---
<img src="https://github.com/bloodmage1/Insam_Decipher/blob/main/Demonstration/load_image.png"/>

다운받은 이미지에서 인삼 x-ray 사진을 불러올 수 있습니다.

---

<img src="https://github.com/bloodmage1/Insam_Decipher/blob/main/Demonstration/loaded_image.png"/>

품질을 가진 인삼 데이터를 불러온 화면입니다.

---

<img src="https://github.com/bloodmage1/Insam_Decipher/blob/main/Demonstration/predicted_image.png"/>

predict 버튼을 클릭하면 인삼의 품질을 확인할 수 있다. 불러 온 사진에서는 해당 x-ray 사진 속 인삼의 등급이 최하인 것을 알 수 있습니다.

## 2. 개발환경

- Window OS, Window 11
- Python 3.12.4
- PySide6

## 3. 디렉토리 구조

```
Insam_decipher/
│
├── main.py
├── ui/
│   ├── main_screen.py
│   ├── describe_screen.py
│   ├── picture_screen.py
│   └── controller_screen.py
└── utils/
    ├── model.py

```

## 4. 각 함수의 기능 설명

### InsamDecipher 클래스

- def main_screen(self)
  - 메인 화면 위젯을 생성하고, Picture_Screen과 Controller_Screen을 레이아웃에 추가합니다.
  - 설정된 레이아웃을 반환합니다.

- def Load_image(self)
  - 파일 다이얼로그를 통해 이미지를 불러오고, 이미지의 너비, 높이, 채널 정보를 업데이트합니다.
  - 이미지를 Matplotlib FigureCanvas에 표시합니다.

- def Predict_image(self)
  - 이미지를 모델에 입력하여 인삼의 상태를 예측하고, 예측된 결과를 설명 라벨에 업데이트합니다.
  
- def Delete_image(self)
  - 현재 표시된 이미지를 삭제하고, 설명 라벨과 사진 정보 라벨을 초기화합니다.

- next_tip(self) <- 분위기 환기용으로 만듬(없어도 됨)
  - 현재 팁 인덱스를 증가시키고, 팁 라벨에 다음 팁을 업데이트합니다. 팁 리스트의 끝에 도달하면 처음으로 돌아갑니다.

### PictureScreen 클래스
  - 사진 화면 위젯을 생성하고, 이미지 설명과 실제 이미지 화면을 추가합니다.
  - 설정된 레이아웃을 반환합니다.

- def image_description(self)
  - 인삼의 상태를 설명하는 라벨을 생성하고 반환합니다.
  
- def Pimage_screen(self)
  - Matplotlib FigureCanvas를 사용하여 이미지를 표시할 화면을 생성하고 반환합니다.

### ControllerScreen 클래스
  - 제어 화면 위젯을 생성하고, 이미지 로드, 인삼 관리, 기능 탭을 추가합니다.
  - 설정된 레이아웃을 반환합니다.

- def Load_Image_screen(self)
  - 이미지 로드 화면 위젯을 생성하고, 파일 라벨, 예측 버튼, 삭제 버튼을 추가합니다.
  - 설정된 레이아웃을 반환합니다.

- def Management_Insam(self)
  - 인삼 정보를 표시하는 그룹박스를 생성하고, 효능, 종류, 수명, 재배 정보를 라벨로 추가합니다.
  - 설정된 레이아웃을 반환합니다.

- def Features_Tap(self) <- 인삼에 대한 정보 소개란(없어도 됨)
  - 탭 위젯을 생성하고, 식물, 꽃, 열매에 대한 정보를 각각의 탭에 추가합니다.
  - 설정된 레이아웃을 반환합니다.

### DescribeScreen 클래스
  - 설명 화면 위젯을 생성하고, 사진 정보와 팁 화면을 추가합니다.
  - 설정된 레이아웃을 반환합니다.

- def Picture_shape_Screen(self)
  - 사진의 너비, 높이, 채널 정보를 표시하는 그룹박스를 생성합니다.
  - 설정된 레이아웃을 반환합니다.

- def Tip_Screen(self) <- 분위기 환기용으로 만듬(없어도 됨)
  - 팁 화면 위젯을 생성하고, 다음 팁으로 넘어가는 버튼과 팁 라벨을 추가합니다.
  - 설정된 레이아웃을 반환합니다.



## 4. 학습결과

Yolov5를 사용하여 69.5%의 결과를 찾은 것을 확인하고, 왜 분류모델에 객체 탐지에 적합한 Yolo를 쓰는 것보다 최신 분류모델을 쓰는 것이 더 결과가 잘 나올 것 같아, 직접 사용해 보고 그 것을 쉽게 확인할 수 있도록 PySide에 담았다.

생각보다 육안으로 확인해보아도 이미지에 유별난 특징을 찾기 힘들어, 이미지의 크기를 256X256 까지 줄이는 대신 batch_size를 늘리고 학습률은 줄여 학습시간을 단축시켰다. 하지만 학습 정확도는 85.9% 까지 늘릴 수 있었다.



## 5. 데이터
[https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71319] 이 곳에서 데이터를 확인하실 수 있습니다.
