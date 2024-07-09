from PySide6.QtWidgets import *

button_size_x = 180
button_size_y = 40
hover_style = """
    QPushButton:hover {
        color: #FFFFFF;
        font-weight: bold;}
"""

class DescribeScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent 
        self.initUI()

    def initUI(self):
        describe_layout = QHBoxLayout(self)

        describe_layout.addWidget(self.PictureShapeScreen(), 2)
        describe_layout.addWidget(self.TipScreen(), 3)

        self.setLayout(describe_layout)

    def PictureShapeScreen(self):
        # self.Picture_width = QLabel('Width: -', self)
        # self.Picture_width.setWordWrap(True)

        # self.Picture_height = QLabel('Height: -', self)
        # self.Picture_height.setWordWrap(True)

        # self.Picture_channel = QLabel('Channels: -', self)
        # self.Picture_channel.setWordWrap(True)

        self.Picture_width = self.parent.Picture_width
        self.Picture_height = self.parent.Picture_height
        self.Picture_channel = self.parent.Picture_channel

        self.shape_group = QGroupBox('사진 정보')

        shape_layout = QGridLayout()
        shape_layout.addWidget(self.Picture_width, 0, 0)
        shape_layout.addWidget(self.Picture_height, 0, 1)
        shape_layout.addWidget(self.Picture_channel, 0, 2)
        self.shape_group.setLayout(shape_layout)

        return self.shape_group

    def TipScreen(self):
        self.Tip_widget = QWidget(self)

        self.next_button = QPushButton('다음', self)
        self.next_button.setFixedSize(button_size_x, button_size_y)
        self.next_button.setStyleSheet(hover_style)
        self.next_button.clicked.connect(self.parent.next_tip)

        # self.Tip_label = QLabel("Tip: -")
        # self.Tip_label.setWordWrap(True)
        self.Tip_label = self.parent.Tip_label 

        Tip_layout = QVBoxLayout(self.Tip_widget)
        Tip_layout.addWidget(self.Tip_label)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.next_button)

        Tip_layout.addLayout(button_layout)
        self.Tip_widget.setLayout(Tip_layout)

        return self.Tip_widget