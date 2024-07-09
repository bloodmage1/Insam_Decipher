from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QHBoxLayout, QScrollArea, QWidget, QLabel, QMessageBox, QFileDialog
from PySide6.QtGui import QIcon, QAction
from qt_material import apply_stylesheet
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from ui.picture_screen import PictureScreen
from ui.controller_screen import ControllerScreen
from ui.describe_screen import DescribeScreen
from utils.predict_image import predict_image, annotate_image
from PIL import Image
import cv2

button_size_x = 180
button_size_y = 40
hover_style = """
    QPushButton:hover {
        color: #FFFFFF;
        font-weight: bold;}
"""

class InsamDecipher(QMainWindow):
    def __init__(self, app, model, transform):
        super().__init__()
        self.app = app
        self.model = model
        self.transform = transform
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

        self.Picture_width = QLabel('Width: -', self)
        self.Picture_height = QLabel('Height: -', self)
        self.Picture_channel = QLabel('Channels: -', self)
        self.description_label = QLabel("이 인삼의 상태는 - 입니다.")
        self.Tip_label = QLabel("Tip: -")
        self.This_is_screen = FigureCanvas(Figure(figsize=(5, 3)))

        self.UI초기화()

    def UI초기화(self):
        Main_box = QVBoxLayout()
        Main_box.addStretch(1)
        Main_box.addWidget(self.main_screen(), 12)
        Main_box.addWidget(DescribeScreen(self), 1)

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
        Load_RGB_menu.triggered.connect(self.load_image)
        fileMenu.addAction(Load_RGB_menu)

        apply_stylesheet(self.app, theme='dark_purple.xml')
        self.setWindowTitle('인삼품질 판독기기')
        self.setWindowIcon(QIcon('./app_img/Title.png'))
        self.setGeometry(0, 0, 1000, 700)
        self.show()

    def main_screen(self):
        main_screen_widget = QWidget(self)
        main_screen_layout = QHBoxLayout(main_screen_widget)
        main_screen_layout.addWidget(PictureScreen(self), 5)
        main_screen_layout.addWidget(ControllerScreen(self), 1)
        main_screen_widget.setLayout(main_screen_layout)
        return main_screen_widget

    def load_image(self):
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
            self.This_is_screen.figure.clear()
            ax = self.This_is_screen.figure.add_subplot(111)
            ax.axis('off')
            ax.imshow(self.image_file_rgb)
            self.This_is_screen.draw()
            self.image_displayed_rgb = True
        else:
            QMessageBox.warning(self, "알림", "이것은 이미지 파일이 아닙니다!!")

    def predict_image(self):
        predicted_label = predict_image(self.model, self.transform, self.image_file_rgb) 
        self.image_file_rgb = annotate_image(self.image_file_rgb, predicted_label) 
        self.description_label.setText(f"이 인삼의 상태는 {predicted_label} 입니다.") 
        
        self.This_is_screen.figure.clear()
        ax = self.This_is_screen.figure.add_subplot(111)
        ax.axis('off')
        ax.imshow(self.image_file_rgb)
        self.This_is_screen.draw()

    def delete_image(self):
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