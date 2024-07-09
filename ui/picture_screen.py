from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class PictureScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.UI초기화(parent)

    def UI초기화(self, parent):
        Picture_Screen_layout = QVBoxLayout(self)

        # self.description_label = QLabel("이 인삼의 상태는 - 입니다.")
        parent.description_label.setWordWrap(True)
        parent.description_label.setMinimumWidth(400)

        # Picture_Screen_layout.addWidget(self.description_label, 1)
        Picture_Screen_layout.addWidget(parent.description_label, 1)

        fig1 = Figure()
        fig1.tight_layout(pad=0)
        fig1.set_facecolor('#28282B')
        # self.This_is_screen = FigureCanvas(fig1)
        parent.This_is_screen = FigureCanvas(fig1)

        # Picture_Screen_layout.addWidget(self.This_is_screen, 10)
        Picture_Screen_layout.addWidget(parent.This_is_screen, 10)
        self.setLayout(Picture_Screen_layout)