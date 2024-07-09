from PySide6.QtWidgets import *
from PySide6.QtGui import QFont
# from ui.load_image_screen import LoadImageScreen
# from ui.management_insam import ManagementInsam
# from ui.features_tab import FeaturesTab

button_size_x = 180
button_size_y = 40
hover_style = """
    QPushButton:hover {
        color: #FFFFFF;
        font-weight: bold;}
"""

class ControllerScreen(QWidget):
    # def __init__(self, parent=None):
    def __init__(self, parent):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        Controller_layout = QVBoxLayout(self)

        Controller_layout.addWidget(self.Load_Image_screen(), 2)
        Controller_layout.addWidget(self.Management_Insam(), 2)
        Controller_layout.addWidget(self.Features_Tap(), 3)

        self.setLayout(Controller_layout)

    def Load_Image_screen(self):
        Load_Image_widget = QWidget(self)

        self.Load_Image_label = QLabel('File', self)
        self.Load_Image_label.setFont(QFont('Helvetica',pointSize=15, weight=1))

        self.Predict_button = QPushButton('Predict', self)
        self.Predict_button.setFixedSize(button_size_x, button_size_y) 
        self.Predict_button.setStyleSheet(hover_style)
        # self.Predict_button.clicked.connect(self.parent().parent().predict_image)
        self.Predict_button.clicked.connect(self.parent().predict_image)

        self.Delete_button = QPushButton('Delete', self)
        self.Delete_button.setFixedSize(button_size_x, button_size_y) 
        self.Delete_button.setStyleSheet(hover_style)
        # self.Delete_button.clicked.connect(self.parent().parent().delete_image)
        self.Delete_button.clicked.connect(self.parent().delete_image)

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
        Features_widget = QWidget()  # 탭을 담을 위젯
        tab_widget = QTabWidget()  # QTabWidget 인스턴스 생성
        Features_layout = QVBoxLayout(Features_widget)  # 메인 레이아웃 설정
        ##
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