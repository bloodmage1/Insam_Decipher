import sys
import warnings
import torch
from PySide6.QtWidgets import *
from ui.main_screen import InsamDecipher
from utils.model import EfficientModel, mytransform

warnings.simplefilter("ignore", UserWarning)
sys.coinit_flags = 2

model = EfficientModel(4).to('cuda')
model.load_state_dict(torch.load('./best_model.pth'))
model.eval()

app = QApplication(sys.argv)
execute_instance = InsamDecipher(app, model, mytransform)
app.exec()