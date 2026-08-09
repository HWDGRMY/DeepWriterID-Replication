import torch
import torch.nn as nn

class DCNN(nn.Module):
    def __init__(self, num_classes=1019):
        super(DCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 80, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(80, 160, kernel_size=2, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(160, 240, kernel_size=2, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(240, 320, kernel_size=2, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(320, 400, kernel_size=2, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        # 🟢 启用 Dropout，防止过拟合，强度 0.2
        self.fc1 = nn.Sequential(
            nn.Linear(1600, 480),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(480, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.output = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.output(x)
        return x