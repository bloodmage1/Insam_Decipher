# Insam Decipher

## 1. Demonstration

This repository contains a tutorial of Insam Quality Classifition using Pytorch.

<img src="https://github.com/bloodmage1/Insam_Decipher/blob/main/Demonstration/play_video.gif"/>

It is a video that briefly shows the quality of 'Insam'.

<img src="https://github.com/bloodmage1/Insam_Decipher/blob/main/Demonstration/insam_decipher_homescreen.png"/>

This is the first screen of the impression Insam Decipher that runs when you run the application.

---
<img src="https://github.com/bloodmage1/Insam_Decipher/blob/main/Demonstration/load_image.png"/>

You can import a Insam x-ray photo from the downloaded image. It is Insam with the lowest(choiha), low(ha), middle(jung), and upper quality(sang), respectively.

---

<img src="https://github.com/bloodmage1/Insam_Decipher/blob/main/Demonstration/loaded_image.png"/>

This is the screen that imported Insam image.

---

<img src="https://github.com/bloodmage1/Insam_Decipher/blob/main/Demonstration/predicted_image.png"/>

You can check the quality of Insam by clicking the 'predict' button. The imported photo shows that the Insam in the x-ray photo is the lowest.

## 2. Prepare Data

Clone this repository with git clone https://github.com/bloodmage1/Insam_Decipher.git.

Video file is in the './img'

## 3. Setup the Environment

The OS is ubuntu-18.04. A Dockerfile with all dependencies is provided. You can build it with

```
docker build -t your_container:your_tag .
```

## 4. Prepare Model

1. Connect to Model
```
docker pull yongyongdie/my_insam_model:latest
```

2. Pull Model
```
docker cp your_container_name:/app/best_model.pth ./model_path_where_you_want
```

3. Setting
```
docker run -it -v /your_predicted_video:/app/test_result your_container
```

4. Enabling Python Environments

```
source Insam_cls/bin/activate
```

## 5. Running Insam Decipher

Quality classification can be performed using the GUI program(Insam Decipher). As shown in 1, an image file is imported and prediction proceeds.

## 6. Development Environment

- Window OS, Window 11
- Python 3.8.7
- PySide6

## 7. Directory Structure

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

## 8. 각 함수의 기능 설명

### InsamDecipher

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

### PictureScreen
  - 사진 화면 위젯을 생성하고, 이미지 설명과 실제 이미지 화면을 추가합니다.
  - 설정된 레이아웃을 반환합니다.

- def image_description(self)
  - 인삼의 상태를 설명하는 라벨을 생성하고 반환합니다.
  
- def Pimage_screen(self)
  - Matplotlib FigureCanvas를 사용하여 이미지를 표시할 화면을 생성하고 반환합니다.

### ControllerScreen
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

### DescribeScreen
  - 설명 화면 위젯을 생성하고, 사진 정보와 팁 화면을 추가합니다.
  - 설정된 레이아웃을 반환합니다.

- def Picture_shape_Screen(self)
  - 사진의 너비, 높이, 채널 정보를 표시하는 그룹박스를 생성합니다.
  - 설정된 레이아웃을 반환합니다.

- def Tip_Screen(self) <- 분위기 환기용으로 만듬(없어도 됨)
  - 팁 화면 위젯을 생성하고, 다음 팁으로 넘어가는 버튼과 팁 라벨을 추가합니다.
  - 설정된 레이아웃을 반환합니다.



## 9. Reusults

I checked that I found 69.5% of the results using Yolov5, and I think it would be better to use the latest classification model than to use Yolo because it is suitable for object detection. so I put it on PySide so that I can use it myself and check it easily.

It was harder than expected to find unusual features in the image even when visually inspected by my eyes. so instead of reducing the image size to 256x256, we shortened the learning time by increasing the batch size and reducing the learning rate. However, the learning accuracy could be increased to 85.9%

## 10. OpenSource Data
[https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71319] 

## 11. Errors I encountered

If an error occurs, please contact us via email.

breakprejudice@naver.com