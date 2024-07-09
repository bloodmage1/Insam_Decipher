import torch
from torch import nn
from efficientnet_pytorch import EfficientNet

from torchvision import transforms

mytransform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class EfficientModel(nn.Module):
    def __init__(self, class_n, rate=0.2):
        super(EfficientModel, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b7')
        self.dropout = nn.Dropout(rate)
        self.output_layer = nn.Linear(in_features=1000, out_features=class_n, bias=True)

    def forward(self, inputs):
        output = self.output_layer(self.dropout(self.model(inputs)))
        return output