import sys
sys.coinit_flags = 2

from PySide6.QtWidgets import *
from PySide6.QtGui import *
from PySide6.QtCore import *

from qt_material import apply_stylesheet

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image

from time import time, sleep
import cv2
import numpy as np

import warnings

warnings.simplefilter("ignore", UserWarning)
sys.coinit_flags = 2

button_size_x = 180
button_size_y = 40
edit_line_size = 120
c_size = 30

hover_style = """
    QPushButton:hover {
        color: #FFFFFF;
        font-weight: bold;}
"""

import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision import transforms

class Efficient_Model(nn.Module):
    def __init__(self, class_n, rate=0.2):
        super(Efficient_Model, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b7')
        self.dropout = nn.Dropout(rate)
        self.output_layer = nn.Linear(in_features=1000, out_features=class_n, bias=True)

    def forward(self, inputs):
        output = self.output_layer(self.dropout(self.model(inputs)))
        return output

mytransform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 모델 불러오기
model = Efficient_Model(4).to('cuda')
model.load_state_dict(torch.load('./best_model.pth'))
model.eval()



class Insam_decipher(QMainWindow):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.current_tip_index = 0 
        self.tips = [
            "Tip: 여운형 선생님은 개성 인삼을 팔아 큰 이윤을 남겨 독립운동 자금으로 사용했습니다.",
            "Tip: 윤봉길 의사님은 상하이에서 인삼 행상을 하며 때를 기다렸습니다.",
            "Tip: 상하이의 인삼 상회는 독립운동의 거점이었습니다.",
            "Tip: 박재혁 의열단원은 싱가포르에서 인삼 상인일을 했습니다.",
            "Tip: 고대로부터 인삼은 한 번 끓인 뒤에 말리는 건삼의 형태로 유통되었지만 훗날 쪄서 건조시키는 증포의 방법으로 홍삼을 만들게 되었습니다.",
            "Tip: 1681년 영국 왕립학회의 주요 수집품 목록에 이미 인삼 뿌리가 수록되어 있었습니다.",
            "Tip: 프랑스에서는 루이 14세 때, 태국에서 찾아 온 사절단이 인삼을 가져오면서 인삼을 알릴 수 있었습니다.",
            "Tip: 불평등 조약이 시작되어, 열강들의 요구에도 다른 물품이 5%의 관세를 매길 때, 홍삼은 15%의 관세를 매겼습니다.",
            "Tip: 외국인은 개인적으로 조선에서 홍삼을 구입할 수 없었습니다.",
        ]
        self.UI초기화()

    def UI초기화(self):
        Main_box = QVBoxLayout()

        Main_box.addStretch(1)
        Main_box.addWidget(self.main_screen(), 12)
        Main_box.addWidget(self.describe_screen(), 1)

        self.scrollArea = QScrollArea()
        self.setCentralWidget(self.scrollArea)

        central_widget = QWidget()
        central_widget.setLayout(Main_box)
        self.scrollArea.setWidget(central_widget)
        self.scrollArea.setWidgetResizable(True) 
    
        menumenu = self.menuBar()
        menumenu.setNativeMenuBar(False)
        fileMenu = menumenu.addMenu('File')

        Load_RGB_menu = QAction('이미지 불러오기', self)
        Load_RGB_menu.setShortcut('Ctrl+R')
        Load_RGB_menu.triggered.connect(self.Load_image)
        fileMenu.addAction(Load_RGB_menu)

        apply_stylesheet(self.app, theme='dark_purple.xml')
        self.setWindowTitle('인삼품질 판독기기')
        self.setWindowIcon(QIcon('Title.png'))
        self.setGeometry(0, 0, 1000, 700)
        self.show()

    def main_screen(self):
        main_screen_widget = QWidget(self)

        main_screen_layout = QHBoxLayout(main_screen_widget)

        main_screen_layout.addWidget(self.Picture_Screen(),5)
        main_screen_layout.addWidget(self.Controller_Screen(),1)

        main_screen_widget.setLayout(main_screen_layout)
        return main_screen_widget

    def Picture_Screen(self):
        Picture_Screen_widget = QWidget(self)
        Picture_Screen_layout = QVBoxLayout(Picture_Screen_widget)

        Picture_Screen_layout.addWidget(self.image_description(), 1)
        Picture_Screen_layout.addWidget(self.Pimage_screen(), 10)

        Picture_Screen_widget.setLayout(Picture_Screen_layout)
        return Picture_Screen_widget
    
    def image_description(self):
        self.description_label = QLabel("이 인삼의 상태는 - 입니다.")
        self.description_label.setWordWrap(True)
        self.description_label.setMinimumWidth(400) 
        return self.description_label

    def Pimage_screen(self):
        fig1 = Figure()
        fig1.tight_layout(pad=0)
        fig1.set_facecolor('#28282B')
        self.This_is_screen = FigureCanvas(fig1)
        return self.This_is_screen
    
    def Controller_Screen(self):
        Controller_widget = QWidget(self)

        Controller_layout = QVBoxLayout(Controller_widget)

        Controller_layout.addWidget(self.Load_Image_screen(), 2)
        Controller_layout.addWidget(self.Management_Insam(), 2)
        Controller_layout.addWidget(self.Features_Tap(), 3)

        Controller_widget.setLayout(Controller_layout)
        return Controller_widget
    
    def Load_Image_screen(self):
        Load_Image_widget = QWidget(self)

        self.Load_Image_label = QLabel('File', self)
        self.Load_Image_label.setFont(QFont('Helvetica',pointSize=15, weight=1))

        self.Predict_button = QPushButton('Predict', self)
        self.Predict_button.setFixedSize(button_size_x, button_size_y) 
        self.Predict_button.setStyleSheet(hover_style)
        self.Predict_button.clicked.connect(self.Predict_image)

        self.Delete_button = QPushButton('Delete', self)
        self.Delete_button.setFixedSize(button_size_x, button_size_y) 
        self.Delete_button.setStyleSheet(hover_style)
        self.Delete_button.clicked.connect(self.Delete_image)

        AGV_state_layout = QVBoxLayout(Load_Image_widget)
        AGV_state_layout.addWidget(self.Load_Image_label)
        AGV_state_layout.addWidget(self.Predict_button)
        AGV_state_layout.addWidget(self.Delete_button)

        Load_Image_widget.setLayout(AGV_state_layout)
        return Load_Image_widget


    def Management_Insam(self):
        self.Management_group = QGroupBox('인삼 정보')

        self.Effect_Insam = QLabel('효능: 심신안정', self)
        self.Effect_Insam.setWordWrap(True)

        self.Type_Insam = QLabel('종류: 인삼속 인삼종', self)
        self.Type_Insam.setWordWrap(True)

        self.Life_Insam = QLabel('수명: 6년', self)
        self.Life_Insam.setWordWrap(True)

        self.Farm_Insam = QLabel('재배: 10~11월', self)
        self.Farm_Insam.setWordWrap(True)

        Management_layout = QGridLayout()
        Management_layout.addWidget(self.Effect_Insam, 0, 0)  
        Management_layout.addWidget(self.Type_Insam, 1, 0) 
        Management_layout.addWidget(self.Life_Insam, 2, 0) 
        Management_layout.addWidget(self.Farm_Insam, 3, 0) 
        self.Management_group.setLayout(Management_layout)

        return self.Management_group
    
    def Features_Tap(self):
        # Features_widget = QWidget() 
        tab_widget = QTabWidget() 
        # Features_layout = QVBoxLayout(Features_widget)  
        grown_plant_tab = QWidget()
        grown_plant_layout = QVBoxLayout()
        height_label = QLabel('식물 높이')
        height_input = QLineEdit('60cm')
        diameter_label = QLabel('지름')
        diameter_input = QLineEdit('대미, 중미, 세미로 구분')
        leaf_color_label = QLabel('잎 색깔')
        leaf_color_input = QLineEdit('녹색')

        grown_plant_layout.addWidget(height_label)
        grown_plant_layout.addWidget(height_input)
        grown_plant_layout.addWidget(diameter_label)
        grown_plant_layout.addWidget(diameter_input)
        grown_plant_layout.addWidget(leaf_color_label)
        grown_plant_layout.addWidget(leaf_color_input)
        grown_plant_tab.setLayout(grown_plant_layout)
        ##
        flower_tab = QWidget()
        grown_plant_layout = QVBoxLayout()
        height_label = QLabel('꽃의 크기')
        height_input = QLineEdit('소형')
        diameter_label = QLabel('암술 수술')
        diameter_input = QLineEdit('수술: 5개, 암술대: 2개')
        leaf_color_label = QLabel('꽃의 색깔')
        leaf_color_input = QLineEdit('연한 녹색')

        grown_plant_layout.addWidget(height_label)
        grown_plant_layout.addWidget(height_input)
        grown_plant_layout.addWidget(diameter_label)
        grown_plant_layout.addWidget(diameter_input)
        grown_plant_layout.addWidget(leaf_color_label)
        grown_plant_layout.addWidget(leaf_color_input)
        flower_tab.setLayout(grown_plant_layout)
        ##
        fruit_tab = QWidget()
        grown_plant_layout = QVBoxLayout()
        height_label = QLabel('열매 크기')
        height_input = QLineEdit('납작한 구형, 작고 여러개')
        diameter_label = QLabel('열매의 맛')
        diameter_input = QLineEdit('인삼향이 나고 매운 맛')
        leaf_color_label = QLabel('수확')
        leaf_color_input = QLineEdit('1년에 1회 채종')

        grown_plant_layout.addWidget(height_label)
        grown_plant_layout.addWidget(height_input)
        grown_plant_layout.addWidget(diameter_label)
        grown_plant_layout.addWidget(diameter_input)
        grown_plant_layout.addWidget(leaf_color_label)
        grown_plant_layout.addWidget(leaf_color_input)
        fruit_tab.setLayout(grown_plant_layout)

        ##
        tab_widget.addTab(grown_plant_tab, "식물")
        tab_widget.addTab(flower_tab, "꽃")
        tab_widget.addTab(fruit_tab, "열매")

        # Features_layout.addWidget(grown_plant_tab)
        # Features_widget.setLayout(Features_layout)

        return tab_widget
    
    def describe_screen(self):
        describe_widget = QWidget(self)

        describe_layout = QHBoxLayout(describe_widget)

        describe_layout.addWidget(self.Picture_shape_Screen(),2)
        describe_layout.addWidget(self.Tip_Screen(),3)

        describe_widget.setLayout(describe_layout)
        return describe_widget
    
    def Picture_shape_Screen(self):
        self.Picture_width = QLabel('Width: -', self)
        self.Picture_width.setWordWrap(True)

        self.Picture_height = QLabel('Height: -', self)
        self.Picture_height.setWordWrap(True)

        self.Picture_channel = QLabel('Channels: -', self)
        self.Picture_channel.setWordWrap(True)

        self.shape_group = QGroupBox('사진 정보')
        
        shape_layout = QGridLayout()  
        shape_layout.addWidget(self.Picture_width, 0, 0)  
        shape_layout.addWidget(self.Picture_height, 0, 1) 
        shape_layout.addWidget(self.Picture_channel, 0, 2) 
        self.shape_group.setLayout(shape_layout)

        return self.shape_group
    
    def Tip_Screen(self):
        self.Tip_widget = QWidget(self)

        self.next_button = QPushButton('다음', self)
        self.next_button.setFixedSize(button_size_x, button_size_y) 
        self.next_button.setStyleSheet(hover_style)
        self.next_button.clicked.connect(self.next_tip)

        self.Tip_label = QLabel("Tip: -")
        self.Tip_label.setWordWrap(True)

        Tip_layout = QVBoxLayout(self.Tip_widget)
        Tip_layout.addWidget(self.Tip_label)

        button_layout = QHBoxLayout()
        button_layout.addStretch()  
        button_layout.addWidget(self.next_button)  

        Tip_layout.addLayout(button_layout)  

        self.Tip_widget.setLayout(Tip_layout)

        return self.Tip_widget
    
####### 함수 부분 ###########
    def Load_image(self):
        '''
        RGB IMAGE 불러오기를 클릭하여 
        파일 확장자명이 .jpg이면 This_is_rgb_screen에 jpg 이미지를 띄운다.
        '''
        self.Image_name = QFileDialog.getOpenFileName(self, 'Open file', './')

        if self.Image_name[0].endswith('.png') or self.Image_name[0].endswith('.jpg'):
            self.image_file_rgb = cv2.imread(self.Image_name[0])
            self.Picture_width.setText(f"Width: {self.image_file_rgb.shape[1]}")
            self.Picture_height.setText(f"Height: {self.image_file_rgb.shape[0]}")
            self.Picture_channel.setText(f"Channel: {self.image_file_rgb.shape[2]}")

            if self.image_file_rgb is None:
                QMessageBox.warning(self, "알림", "이미지를 불러올 수 없습니다. 파일 경로를 확인하세요.")
                return
            
            self.image_file_rgb = cv2.cvtColor(self.image_file_rgb, cv2.COLOR_BGR2RGB)
            self.image_file_rgb = Image.fromarray(self.image_file_rgb)
            
            QLabel('Width: -', self)

            self.This_is_screen.figure.clear() 
            ax = self.This_is_screen.figure.add_subplot(111)
            ax.axis('off')
            ax.imshow(self.image_file_rgb)
            self.This_is_screen.draw()
            self.image_displayed_rgb = True
        else:
            QMessageBox.warning(self, "알림", "이것은 이미지 파일이 아닙니다!!")

    def Predict_image(self):
        image = mytransform(self.image_file_rgb).unsqueeze(0).to('cuda')

        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            label_mapping = {0: '상', 1: '중', 2: '하', 3: '최하'}
            predicted_label = label_mapping[predicted.item()]

        self.description_label.setText(f"이 인삼의 상태는 {predicted_label} 입니다.")

    def Delete_image(self):
        self.This_is_screen.figure.clear()
        self.description_label.setText(f"이 인삼의 상태는 - 입니다.")
        self.Picture_width.setText(f"Width: -")
        self.Picture_height.setText(f"Height: -")
        self.Picture_channel.setText(f"Channel: -")

        ax = self.This_is_screen.figure.add_subplot(111)
        ax.axis('off')
        self.This_is_screen.draw()

    def next_tip(self):
        self.current_tip_index += 1  
        if self.current_tip_index >= len(self.tips):  
            self.current_tip_index = 0
        self.Tip_label.setText(self.tips[self.current_tip_index]) 

app = QApplication(sys.argv)
execute_instance = Insam_decipher(app)
app.exec()
