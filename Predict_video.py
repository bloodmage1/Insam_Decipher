import sys
sys.coinit_flags = 2

from PySide6.QtWidgets import *
from PySide6.QtGui import *
from PySide6.QtCore import *

from qt_material import apply_stylesheet

from time import time, sleep
import os
import cv2
import warnings

import torch

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


class YOLO_Object_Detector:
    def __init__(self, model_name='yolov5s'):
        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)

    def detect_objects_in_video_and_save_frames(self, video_path, output_dir="./testtest", frame_rate=10):
        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame)
            detections = results.xyxy[0].cpu().numpy()  

            for *xyxy, conf, cls in detections:
                if results.names[int(cls)] == 'car': 
                    label = f'{results.names[int(cls)]} {conf:.2f}'
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            frame_filename = os.path.join(output_dir, f'frame_{frame_count:04d}.png')
            cv2.imwrite(frame_filename, frame)
            frame_count += 1

        cap.release()
        images = [img for img in os.listdir(output_dir) if img.endswith(".png")]
        images.sort() 

        first_frame_path = os.path.join(output_dir, images[0])
        frame = cv2.imread(first_frame_path)
        height, width, layers = frame.shape

        frame = cv2.imread(first_frame_path)
        height, width, layers = frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        video_writer = cv2.VideoWriter("predicted_video.mp4", fourcc, frame_rate, (width, height))
        for image in images:
            img_path = os.path.join(output_dir, image)
            frame = cv2.imread(img_path)
            video_writer.write(frame)

        video_writer.release()

class Find_Car_in_blackbox(QMainWindow):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.video_path = None  
        self.cap = None
        self.cap_p = None
        self.UI초기화()

    def UI초기화(self):
        Main_box = QHBoxLayout()

        Main_box.addStretch(1)
        Main_box.addWidget(self.Original_Video(), 7)
        Main_box.addWidget(self.Predict_Video(), 7)
        Main_box.addWidget(self.Controller(), 1)

        self.scrollArea = QScrollArea()
        self.setCentralWidget(self.scrollArea)

        central_widget = QWidget()
        central_widget.setLayout(Main_box)
        self.scrollArea.setWidget(central_widget)
        self.scrollArea.setWidgetResizable(True) 

        apply_stylesheet(self.app, theme='light_cyan.xml')
        self.setWindowTitle('차 검출기')
        self.setWindowIcon(QIcon('Title.png'))
        self.setGeometry(0, 0, 1400, 800)
        self.show()

    def Original_Video(self):
        main_screen_widget = QWidget(self)

        main_screen_layout = QVBoxLayout(main_screen_widget)

        self.label = QLabel(self)
        self.label.setGeometry(10, 10, 640, 360) 
        main_screen_layout.addWidget(self.label)

        self.play_button = QPushButton('Play', self)
        self.play_button.clicked.connect(self.play_video)
        main_screen_layout.addWidget(self.play_button)

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        return main_screen_widget
    

    def play_video(self):
        if self.cap:
            self.cap.release()
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            self.timer.start(30)
        else:
            QMessageBox.warning(self, "알림", "비디오 파일이 로드되지 않았습니다.")

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 360)) 
            image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_BGR888)
            self.label.setPixmap(QPixmap.fromImage(image))
        else:
            self.timer.stop()
            self.cap.release()

    def closeEvent(self, event):
        self.cap.release()

    def Predict_Video(self):
        main_screen_widget = QWidget(self)

        main_screen_layout = QVBoxLayout(main_screen_widget)

        self.label_p = QLabel(self)
        self.label_p.setGeometry(10, 10, 640, 360)  
        main_screen_layout.addWidget(self.label_p)

        self.play_button_p = QPushButton('Play', self)
        self.play_button_p.clicked.connect(self.play_video_p)
        main_screen_layout.addWidget(self.play_button_p)

        self.cap_p = None
        self.timer_p = QTimer()
        self.timer_p.timeout.connect(self.update_frame_p)
        return main_screen_widget

    def play_video_p(self):
        if self.cap_p:
            self.cap_p.release()
        self.cap_p = cv2.VideoCapture("predicted_video.mp4")
        self.timer_p.start(30)

    def update_frame_p(self):
        ret, frame = self.cap_p.read()
        if ret:
            frame = cv2.resize(frame, (640, 360)) 
            image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_BGR888)
            self.label_p.setPixmap(QPixmap.fromImage(image))
        else:
            self.timer_p.stop()
            self.cap_p.release()


    def closeEvent(self, event):
        self.cap_p.release()


    def Controller(self):
        Load_Image_widget = QWidget(self)

        self.Load_button = QPushButton('Load File', self)
        self.Load_button.setFixedSize(button_size_x, button_size_y) 
        self.Load_button.setStyleSheet(hover_style)
        self.Load_button.clicked.connect(self.Load_Video)

        self.Predict_button = QPushButton('Predict', self)
        self.Predict_button.setFixedSize(button_size_x, button_size_y) 
        self.Predict_button.setStyleSheet(hover_style)
        self.Predict_button.clicked.connect(self.Predict_image)

        self.Delete_button = QPushButton('Delete', self)
        self.Delete_button.setFixedSize(button_size_x, button_size_y) 
        self.Delete_button.setStyleSheet(hover_style)
        self.Delete_button.clicked.connect(self.Delete_image)

        AGV_state_layout = QVBoxLayout(Load_Image_widget)
        AGV_state_layout.addWidget(self.Load_button)
        AGV_state_layout.addWidget(self.Predict_button)
        AGV_state_layout.addWidget(self.Delete_button)
        AGV_state_layout.addStretch(1)

        Load_Image_widget.setLayout(AGV_state_layout)
        return Load_Image_widget
    

####
    def Load_Video(self):
        self.video_path, _ = QFileDialog.getOpenFileName(self, 'Open file', 
                                                         './', "Video Files (*.mp4 *.avi *.mov)")
        if self.video_path:
            QMessageBox.information(self, "알림", f"비디오 경로가 갱신되었습니다: {self.video_path}")
        else:
            QMessageBox.warning(self, "알림", "비디오 파일을 선택하지 않았습니다.")

    def Predict_image(self):
        if self.video_path:
            detector = YOLO_Object_Detector()
            detector.detect_objects_in_video_and_save_frames(self.video_path)
            QMessageBox.information(self, "알림", "예측이 완료되었습니다.")
        else:
            QMessageBox.warning(self, "알림", "비디오 파일이 로드되지 않았습니다.")


    def Delete_image(self):
        self.label.clear()
        self.label_p.clear()

app = QApplication(sys.argv)
execute_instance = Find_Car_in_blackbox(app)
app.exec()
