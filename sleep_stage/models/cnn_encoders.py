import torch.nn as nn
import torchvision
from torchvision.models import ResNet18_Weights, ResNet50_Weights, RegNet_Y_128GF_Weights, RegNet_Y_16GF_Weights


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # BatchNorm에 bias가 포함되어 있으므로, conv2d는 bias=False로 설정합니다.
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion),
        )

        # identity mapping, input과 output의 feature map size, filter 수가 동일한 경우 사용.
        self.shortcut = nn.Sequential()

        self.relu = nn.ReLU()

        # projection mapping using 1x1conv
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes=5, init_weights=True, num_channels=9):
        super().__init__()
        
        self.in_channels=32
        self.num_classes = num_classes

        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layers = nn.ModuleList()
        self.num_planes = [64, 128, 128, 256]
        self.strides = [1, 2, 2, 2]

        for i in range(0, 4):
            layer = self._make_layer(block, self.num_planes[i], num_block[i], stride=self.strides[i])
            self.layers.append(layer)

        self.c_out = 64
        self.dropout = nn.Dropout(0.5)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256 * block.expansion, num_classes)
                
        # weights inittialization
        if init_weights:
            self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch_size, seq_len, 1500, num_channels)
        if len(x.shape) == 5:
            x = x.squeeze(0)

        bs, seq_len, num_signals, num_channels = x.size()
        seq_len = x.size(1)
        x = x.permute(0, 3, 1, 2)  # (batch_size, num_channels, seq_len, 1500)
        n = 10
        x = x.reshape(bs, num_channels, seq_len * n, num_signals // n)

        out = self.conv1(x) 

        for layer in self.layers:
            out = layer(out)

        x = self.dropout(out)
        x = self.avg_pool(out) # [batch_size, 512, 1, 1]
        x = x.view(x.size(0), -1) # [batch_size, 512]

        x = self.fc(x) # [batch_size, num_classes]

        return x

    # define weight initialization function
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def resnet18(num_channels=9, num_classes=5, pretrained=False):
    if pretrained:
        model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = torchvision.models.resnet18(weights=None)
        
    num_features = model.fc.in_features
    model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(num_features, num_classes)
    return model


def resnet50(num_channels=9, num_classes=5, pretrained=True):
    model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    num_features = model.fc.in_features
    model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

def regnet128(num_channels=9, num_classes=5, pretrained=True):
    model = torchvision.models.regnet_y_128gf(weights=RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_E2E_V1)
    # print(model)
    num_features = model.fc.in_features
    model.stem[0] = nn.Conv2d(num_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

def regnet16(num_channels=9, num_classes=5, pretrained=True):
    model = torchvision.models.regnet_y_16gf(weights=RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_E2E_V1)
    num_features = model.fc.in_features
    model.stem[0] = nn.Conv2d(num_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

def swin_transformer(num_channels=9, num_classes=5, pretrained=True):
    model = torchvision.models.swin_v2_b(weights=torchvision.models.Swin_V2_B_Weights.IMAGENET1K_V1)
    model.features[0][0] = nn.Conv2d(num_channels, 128, kernel_size=(4, 4), stride=(4, 4))
    num_features = model.head.in_features
    model.head = nn.Linear(num_features, num_classes)
    
    return model

def conv_next(num_channels=9, num_classes=5, pretrained=True):
    model = torchvision.models.convnext_base(weights=torchvision.models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
    model.features[0][0] = nn.Conv2d(num_channels, 128, kernel_size=(4, 4), stride=(4, 4), padding=(0, 0), bias=False)
    num_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_features, num_classes)
    
    return model

if __name__ == "__main__":
    # Example usage
    import torch
    model = resnet18(num_channels=9, num_classes=5, pretrained=False)
    x = torch.randn(2, 9, 1, 30000)  # Example input tensor
    print(model(x).shape)  # Should output torch.Size([2, 5])

